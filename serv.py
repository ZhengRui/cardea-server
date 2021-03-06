#!/usr/bin/env python

from gestureModel import gestDetModel
from faceModel import faceRecModel

import socket
import threading
from multiprocessing import Process, JoinableQueue, Event, Value
import Queue
from multiprocessing.dummy import Process as DummyProcess
import SocketServer as SS
import signal
import struct
import numpy as np
import math
import cv2
import sys
import os
import time
import cPickle as pkl
import pandas as pd
from svmutil import *
import caffe


class RequestHandler(SS.BaseRequestHandler):

    #  To subclass, check exmaple:
    #  http://bioportal.weizmann.ac.il/course/python/PyMOTW/PyMOTW/docs/SocketServer/index.html

    def __init__(self, request, client_address, server):
        self.name = threading.currentThread().getName()
        self.res = {}
        skt_clients_map[self.name] = self
        SS.BaseRequestHandler.__init__(self, request, client_address, server)
        return

    def handle(self):
        print "tcp request received, create client ", self.name

        headerBytes = self.request.recv(32)
        header = struct.unpack('8i', headerBytes)
        print header

        if not header[0]:   # frame prediction message
            self.res={'hand_res':None, 'hand_res_evt':Event(), 'face_res':None, 'face_res_evt':Event(),
                        'bbsprofiles':{}, 'c1':[], 'c2':[], 'c3':[], 'c4':[], 'c5':[]}
            # in self.res['bbsprofiles'][i] = [gesture, location, scene, policy, username, otfeatures]

            lat, lon = struct.unpack('<2d', self.request.recv(16))
            #  print lat, lon

            size = header[5]
            mode = header[6]
            size_recv = 0
            frm_buffer = ''

            while size_recv < size:
                chunk = self.request.recv(size - size_recv)
                if not chunk:
                    print "socket receiving error"
                    return None
                else:
                    frm_buffer += chunk
                    size_recv += len(chunk)
                    #  print "size, size_recv", size, size_recv

            if mode:
                size_extra = header[7]
                size_recv = 0
                extra_buffer = ''
                while size_recv < size_extra:
                    chunk = self.request.recv(size_extra - size_recv)
                    if not chunk:
                        print "socket receiving error"
                        return None
                    else:
                        extra_buffer += chunk
                        size_recv += len(chunk)

                facenum = size_extra / (4 * 260)
                facebbxs = np.fromstring(extra_buffer[:16*facenum], dtype=np.int32)
                facefeats = np.fromstring(extra_buffer[16*facenum:], dtype=np.float32)
                if facenum:
                    facebbxs = facebbxs.reshape((facenum, -1))
                    facefeats = facefeats.reshape((facenum, -1))


            print "Frame received"

            #  frm_raw_q.put((header[1], header[2], frm_buffer))

            #  leave the efforts of rotation and flip preprocessing to the socket client,
            #  no need of worker_preprocess_p and frm_raw_q, frm_buffer received here is
            #  already preprocessed by the client, directly put frm_buffer to hand_inp_q
            #  and face_inp_q

            hand_inp_q.put((self.name, frm_buffer))


            if not mode:  #  if weak mode
                face_inp_q.put((self.name, frm_buffer, 0))
            else:  #  if strong mode
                face_inp_q.put((self.name, (facebbxs, facefeats), 1))

            self.res['hand_res_evt'].wait()
            self.res['face_res_evt'].wait()
            print "hand result : ", self.res['hand_res'][:, -2:], "\nface result : ", self.res['face_res'][1], np.max(self.res['face_res'][2], 1) if len(self.res['face_res'][2]) else []


            # Case 0: no registered user, no operation
            # Case 1: gesture 'no', blurring
            # Case 2: gesture 'yes', not blurring
            # Case 3: out of distance, not blurring
            # Case 4: otfeature matched, blurring
            # Case 5: sending scene lists back

            # is any registered user?

            if len(self.res['bbsprofiles']) == 0:
                print "Case 0: no registered user, no operation"

            else: # for registered users
                # find nearest face for each yes or no gesture
                gesture_user = {2:[], 3:[]}

                recs, pps, _, feats = self.res['face_res']
                face_centers = np.zeros((len(recs), 2))
                for i in range(len(recs)):
                    center = recs[i].center()
                    face_centers[i, :] = [center.x, center.y]


                for (x0, y0, x1, y1, score, hand_cls) in self.res['hand_res']:
                    if hand_cls != 1: # not for natural hand
                        hand_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
                        dis_centers = np.linalg.norm(face_centers-hand_center, ord=2, axis=1)
                        gesture_user[hand_cls].append(np.argmin(dis_centers))

                for i, profile in self.res['bbsprofiles'].iteritems():
                    # 'yes' gesture is used
                    if profile[0][0] == 1 and i in gesture_user[2]:
                        print "Case 2: gesture 'yes', not blurring"
                        self.res['c2'].append(i)

                    # 'no' gesture is used
                    elif profile[0][1] == 1 and i in gesture_user[3]:
                        print "Case 1: gesture 'no', blurring"
                        self.res['c1'].append(i)

                    # no hand gesture is triggered
                    else:
                        # location filter
                        distance = calDistance((lat, lon), profile[1])
                        if distance > 2.0:
                            print "Case 3: out of distance, not blurring"
                            self.res['c3'].append(i)

                        else:
                            # compare others features in profiles with those in the picture

                            should_blur = calSimilarity(profile[5], feats)
                            if should_blur:
                                print "Case 4: otfeature matched, blurring"
                                self.res['c4'].append(i)
                            else: # no matched otfeatures
                                # send back scene lists
                                print 'Case 5: sending scene lists back'
                                self.res['c5'].append(i)


            # send message back to the client
            # 4 bytes (face length) | 4 bytes (hand_length) | real data
            # prepare results to send back to the client
            length_face = ''
            length_hand = ''
            real_data = ''
            data_to_send = ''

            # pack all faces
            # 4i (bbs) | 28s (username) | 2s (case) | 10s (scenes) | 1i (policy)
            recs, pps, _, _ = self.res['face_res']

            # pack unregistered users
            registered_bbss = self.res['bbsprofiles'].keys()
            for bbs_idx in range(len(recs)):
                if bbs_idx not in registered_bbss:
                    bbs = transRec(recs[bbs_idx])
                    data = struct.pack('4i28s2s10s1i', bbs[0], bbs[1], bbs[2], bbs[3], fillString('Unknown', 28), 'c0', '', -1)

                    real_data += data

            # pack c1 users: gesture 'no', blurring
            for bbs_idx in self.res['c1']:
                bbs = transRec(recs[bbs_idx])
                policy = self.res['bbsprofiles'][bbs_idx][3]
                data = struct.pack('4i28s2s10s1i', bbs[0], bbs[1], bbs[2], bbs[3], fillString(uid2name[pps[bbs_idx]], 28), 'c1', '', policy)
                real_data += data

            # pack c2 users: gesture 'yes', not blurring
            for bbs_idx in self.res['c2']:
                bbs = transRec(recs[bbs_idx])
                data = struct.pack('4i28s2s10s1i', bbs[0], bbs[1], bbs[2], bbs[3], fillString(uid2name[pps[bbs_idx]], 28), 'c2', '', -1)
                real_data += data

            # pack c3 users: out of distance, not blurring
            for bbs_idx in self.res['c3']:
                bbs = transRec(recs[bbs_idx])
                data = struct.pack('4i28s2s10s1i', bbs[0], bbs[1], bbs[2], bbs[3], fillString(uid2name[pps[bbs_idx]], 28), 'c3', '', -1)
                real_data += data

            # pack c4 users: otfeature matched, blurring
            for bbs_idx in self.res['c4']:
                bbs = transRec(recs[bbs_idx])
                policy = self.res['bbsprofiles'][bbs_idx][3]
                data = struct.pack('4i28s2s10s1i', bbs[0], bbs[1], bbs[2], bbs[3], fillString(uid2name[pps[bbs_idx]], 28), 'c4', '', policy)
                real_data += data

            # pack c5 users: sending scene lists back
            for bbs_idx in self.res['c5']:
                bbs = transRec(recs[bbs_idx])
                policy = self.res['bbsprofiles'][bbs_idx][3]
                scene_list = ['0'] * 10
                for scene in self.res['bbsprofiles'][bbs_idx][2]:
                    if scene >= 0:
                        scene_list[scene] = '1'
                data = struct.pack('4i28s2s10s1i', bbs[0], bbs[1], bbs[2], bbs[3], fillString(uid2name[pps[bbs_idx]], 28), 'c5', ''.join(scene_list), policy)
                real_data += data

            size_face = len(real_data)
            length_face = struct.pack('1i', size_face)

            # pack all hands
            # 4i (rectangles) | 1i (hand class) | 1f (score)
            for x0, y0, x1, y1, scr, hand_cls in self.res['hand_res']:
                data = struct.pack('5i1f', int(x0), int(y0), int(x1), int(y1), int(hand_cls), scr)
                real_data += data

            size_hand = len(real_data) - size_face
            length_hand = struct.pack('1i', size_hand)

            data_to_send = length_face + length_hand + real_data
            self.request.send(data_to_send)

            #cv2.namedWindow("ResultPreview", cv2.CV_WINDOW_AUTOSIZE)
            #nparr = np.fromstring(frm_buffer, dtype=np.uint8)
            #im_show = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
            #for (x0, y0, x1, y1, score, cls) in self.res['hand_res']:
                #cv2.rectangle(im_show, (int(x0), int(y0)), (int(x1), int(y1)), (0,0,255), 2)
                #cv2.putText(im_show, '{:.1f} : {:.3f}'.format(cls, score), (int(x0), int(y0)-10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 2)
            recs, pps, scrs, _ = self.res['face_res']
            for i in range(len(recs)):
                #cv2.rectangle(im_show, (recs[i].left(), recs[i].top()), (recs[i].right(), recs[i].bottom()), (0,255,0), 2)
                #cv2.putText(im_show, '{}'.format(uid2name[pps[i]] if max(scrs[i]) > 0.3 else '****'), (recs[i].left(), recs[i].top()-10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
                print uid2name[pps[i]] if max(scrs[i]) > 0.3 else 'unknown person'
            #im_show = cv2.resize(im_show, (int(im_show.shape[1]/2.), int(im_show.shape[0]/2.)), interpolation=cv2.INTER_AREA)
            #cv2.imshow("ResultPreview", im_show)
            #cv2.waitKey(0)


        else:   # profile message
            self.res={'pref_wrt_evt':Event()}

            size = sum(header[1:])
            size_recv = 0
            pref_buffer = ''

            while size_recv < size:
                chunk = self.request.recv(size - size_recv)
                if not chunk:
                    print "socket receiving error"
                    return None
                else:
                    pref_buffer += chunk
                    size_recv += len(chunk)
                    #  print "size, size_recv", size, size_recv

            print "Preference updation received"

            parser_str = '<' + str(header[1]) + 's' + '2?2d' + str(header[4]/4) + 'i1i'
            pref_misc = struct.unpack(parser_str, pref_buffer[:sum(header[1:-2])])
            pref_myfeat = np.fromstring(pref_buffer[sum(header[1:-2]):sum(header[1:-1])], dtype='<f4').reshape((header[-2]/1024, 256))
            pref_otfeat = np.fromstring(pref_buffer[sum(header[1:-1]):], dtype='<f4').reshape((header[-1]/1024, 256))
            #  print pref_misc, pref_myfeat.shape, pref_otfeat.shape
            pref_wrt_q.put((self.name, pref_misc, pref_myfeat, pref_otfeat))
            self.res['pref_wrt_evt'].wait()
            self.request.send(struct.pack("?", True))


    def finish(self):
        skt_clients_map.pop(self.name)
        print "tcp request finished"
        return SS.BaseRequestHandler.finish(self)

class Server(SS.ThreadingMixIn, SS.TCPServer):
    pass

def modelPrepare(model_class_name, params, tsk_queues):
    #  caffe initialization has to be put here, otherwise encounter:
       #  "syncedmem.cpp: error == cudaSuccess (3 vs. 0)"
    #  or
       #  "math_functions.cu:28: CUBLAS_STATUS_SUCCESS (14 vs. 0) CUBLAS_STATUS_INTERNAL_ERROR"
    caffe.set_mode_cpu()
    modelClass = getattr(sys.modules[__name__], model_class_name)
    model = modelClass(*params)
    model.serv(*tsk_queues)


#  def preProcess(raw_q, inp_qs):
    #  while True:
        #  front_or_back, orient_case, frm_buffer = raw_q.get()
        #  # decode jpeg
        #  nparr = np.fromstring(frm_buffer, dtype=np.uint8)
        #  img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

        #  # preprocessing frame : flip , orientation
        #  if orient_case == 0:
            #  img_np = cv2.transpose(img_np)
            #  img_np = cv2.flip(img_np, 1-front_or_back)
        #  elif orient_case == 1:
            #  img_np = cv2.flip(img_np, -1)
        #  elif orient_case == 2:
            #  img_np = cv2.transpose(img_np)
            #  img_np = cv2.flip(img_np, front_or_back)
        #  else:
            #  pass
        #  print "Frame shape: ", img_np.shape

        #  # encode back for the speed up of IPC through serialization
        #  _, frm_buffer = cv2.imencode(".jpg", img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #  for q in inp_qs:
            #  q.put(frm_buffer)
        #  raw_q.task_done()


def resMailMan(res_q, jobtype): # jobtype is 'hand_res' or 'face_res'
    while dummycontinue:
        try:
            cli_name, res = res_q.get(timeout=2)
            client = skt_clients_map[cli_name]
            client.res[jobtype] = res

            # retrieve registered users' profiles, rewrite myfeature with bbx index
            if jobtype == 'face_res':
                recs, pps, scrs, feats = res
                for i in range(len(recs)):
                    if max(scrs[i]) > 0.3:
                        username = uid2name[pps[i]]
                        client.res['bbsprofiles'][i] = pref_db.ix[username].values.tolist()
                        client.res['bbsprofiles'][i][4] = username

            client.res[jobtype + '_evt'].set()
            #  print cli_name, " ", jobtype, " result mailed"
        except Queue.Empty:
            #  print 'timeout in ', jobtype
            if jobtype == 'face_res':
                faceres_empty_evt.set()
            pass


def savePref(pref_q, jobtype):  # jobtype is 'pref_wrt'
    while dummycontinue:
        try:
            cli_name, pref_misc, pref_myfeat, pref_otfeat = pref_q.get(timeout=2)
            # save profile to pref_add_db and dump pref_add_db to pref_add_fd

            username = pref_misc[0]
            #print username, pref_myfeat.shape, pref_otfeat.shape
            add_wrt_evt.wait()

            if username in pref_add_db.index:
                pref_myfeat_old = pref_add_db.get_value(username, 'myfeature')
                pref_myfeat = np.vstack((pref_myfeat_old, pref_myfeat))
                pref_otfeat = pref_otfeat if pref_otfeat.shape[0] else pref_add_db.get_value(username, 'otfeature')

            pref_add_db.set_value(username, 'gesture', pref_misc[1:3])
            pref_add_db.set_value(username, 'location', pref_misc[3:5])
            pref_add_db.set_value(username, 'scene', pref_misc[5:-1])
            pref_add_db.set_value(username, 'policy', pref_misc[-1])
            pref_add_db.set_value(username, 'myfeature', pref_myfeat)
            pref_add_db.set_value(username, 'otfeature', pref_otfeat)

            client = skt_clients_map[cli_name]
            client.res[jobtype + '_evt'].set()
        except Queue.Empty:
            pass

def mergePref(fresh, tot):
    for username in fresh.index:
        myfeat = fresh.get_value(username, 'myfeature')
        otfeat = fresh.get_value(username, 'otfeature')
        if username in tot.index:
            myfeat = np.vstack((tot.get_value(username, 'myfeature'), myfeat))
            otfeat = otfeat if otfeat.shape[0] else tot.get_value(username, 'otfeature')

        tot.set_value(username, 'gesture', fresh.get_value(username, 'gesture'))
        tot.set_value(username, 'location', fresh.get_value(username, 'location'))
        tot.set_value(username, 'scene', fresh.get_value(username, 'scene'))
        tot.set_value(username, 'policy', fresh.get_value(username, 'policy'))
        tot.set_value(username, 'myfeature', myfeat)
        tot.set_value(username, 'otfeature', otfeat)

def svmTrain(db, ratio = 0.8):
    uid_to_name = {}
    uid = 0

    for username, pref in db.iterrows():
        feat = pref['myfeature']
        if feat.shape[0]:
            uid_to_name[uid] = username
            label = uid * np.ones((feat.shape[0],1))
            idx_rand = np.random.permutation(feat.shape[0])
            pos_split = int(ratio * len(idx_rand))
            train_idx = idx_rand[:pos_split]
            test_idx = idx_rand[pos_split:]
            train_feat = np.vstack((train_feat, feat[train_idx, :])) if uid else feat[train_idx, :]
            train_label = np.vstack((train_label, label[train_idx, :])) if uid else label[train_idx, :]
            test_feat = np.vstack((test_feat, feat[test_idx, :])) if uid else feat[test_idx, :]
            test_label = np.vstack((test_label, label[test_idx, :])) if uid else label[test_idx, :]
            uid += 1

    prob = svm_problem(map(int, train_label.flatten().tolist()), train_feat.tolist())
    param = svm_parameter('-t 0 -c 4 -b 1')
    model = svm_train(prob, param)

    print "test split: "
    svm_predict(map(int, test_label.flatten().tolist()), test_feat.tolist(), model)

    print "total data: "
    svm_predict(map(int, np.vstack((train_label, test_label)).flatten().tolist()), np.vstack((train_feat, test_feat)).tolist(), model)

    return model, uid_to_name

def updateFaceClassifier(interval):
    global pref_add_db, pref_db, uid2name
    while dummycontinue:
        time.sleep(interval)
        if len(pref_add_db.index):
            num_new_feat = 0    # number of users who uploaded new features
            for feat in pref_add_db['myfeature']:
                num_new_feat += 1 if feat.shape[0] else 0

            # trainer takes out new features from pref_add_db to train new face classifier
            add_wrt_evt.clear() # don't write new features to pref_add_db before trainer takes it out
            pref_fresh = pref_add_db
            pref_add_db = pd.DataFrame(index=[], columns=['username', 'gesture', 'location', 'scene', 'policy', 'myfeature', 'otfeature'])
            pref_add_db.set_index(['username'], inplace=True)
            add_wrt_evt.set()

            pref_db_tot = pref_db.copy()
            mergePref(pref_fresh, pref_db_tot)

            with open('./profiles/profiles.pkl', 'wb') as pref_fd:
                pkl.dump(pref_db_tot, pref_fd)

            num_tot_feat = 0    # number of users who has features in the database
            for feat in pref_db_tot['myfeature']:
                num_tot_feat += 1 if feat.shape[0] else 0

            svm_reload = False
            if num_new_feat and num_tot_feat >= 2:  # if in last interval there are users uploaded new features and svm is trainable
                mdl_svm_fresh, uid_to_name = svmTrain(pref_db_tot)
                svm_save_model('./profiles/svm_model.bin', mdl_svm_fresh)
                with open('./profiles/uid2name.pkl', 'wb') as uid2name_fd:
                    pkl.dump(uid_to_name, uid2name_fd)
                svm_reload = True

            # update svm model for face workers and replace pref_db, uid2name in memory
            # with pref_db_tot, uid_to_name
            facemdl_continue_evt.clear() # time to block upstream face workers to process frames
            faceres_empty_evt.wait()  # wait for downstream face results depleted by mailman
            pref_db = pref_db_tot
            if svm_reload:
                uid2name = uid_to_name
                svm_reload_1.value = 1
            facemdl_continue_evt.set()
            faceres_empty_evt.clear()

def calDistance(origin, destination):
    if not len(destination) or (origin[0] == 0 and origin[0] == 0):
        return 0
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d # km

def calSimilarity(otfeats, feats):
    dis_threshold = 100
    ratio = 0.5
    tot = otfeats.shape[0]
    if not tot:
        return False
    else:
        for feat in feats:
            dis = np.linalg.norm(otfeats-feat, ord=2, axis=1)
            if sum(dis <= dis_threshold) / tot >= ratio:
                return True
        return False

def transRec(dlib_rec):
    bbs_x0 = dlib_rec.left()
    bbs_y0 = dlib_rec.top()
    bbs_x1 = dlib_rec.right()
    bbs_y1 = dlib_rec.bottom()
    return bbs_x0, bbs_y0, bbs_x1, bbs_y1

def fillString(s, l):
    ext = l - len(s)
    return s[:l] if ext < 0 else (s + ' ' * ext)

if __name__ == "__main__":

    import wingdbstub
    if 'WINGDB_ACTIVE' in os.environ:
        print "Success starting debug"
    else:
        print "Failed to start debug... Continuing without debug"

    skt_clients_map = {}

    #  frm_raw_q = JoinableQueue()
    hand_inp_q = JoinableQueue()
    hand_res_q = JoinableQueue()
    face_inp_q = JoinableQueue()
    face_res_q = JoinableQueue()
    pref_wrt_q = JoinableQueue()

    add_wrt_evt = Event()
    add_wrt_evt.set()
    faceres_empty_evt = Event()
    facemdl_continue_evt = Event()
    facemdl_continue_evt.set()

    #  worker_preprocess_p = Process(target = preProcess, args = (frm_raw_q, (hand_inp_q, )))

    worker_hand_p1 = Process(target = modelPrepare, args = ('gestDetModel',
    ("../modelzoo/hand/test.prototxt", "../modelzoo/hand/vgg16_faster_rcnn_handGesdet_aug_fulldata_iter_50000.caffemodel", 0.4, 0.8, ('natural', 'yes', 'no')), (hand_inp_q, hand_res_q)))

    svm_reload_1 = Value('i', 0)
    worker_face_p1 = Process(target = modelPrepare, args = ('faceRecModel',
    ("../modelzoo/face/haarcascade_frontalface_alt.xml", "./align/shape_predictor_68_face_landmarks.dat", "../modelzoo/face/LightenedCNN_B_deploy.prototxt", "../modelzoo/face/LightenedCNN_B.caffemodel", "eltwise_fc1", "./profiles/svm_model.bin"), (face_inp_q, face_res_q, facemdl_continue_evt, svm_reload_1, "./profiles/svm_model.bin")))

    dummycontinue = True
    worker_handres_mailman = DummyProcess(target = resMailMan, args = (hand_res_q, 'hand_res'))
    worker_faceres_mailman = DummyProcess(target = resMailMan, args = (face_res_q, 'face_res'))
    worker_pref_writeman = DummyProcess(target = savePref, args = (pref_wrt_q, 'pref_wrt'))
    worker_svm_trainer = DummyProcess(target = updateFaceClassifier, args = (10,))

    #  worker_preprocess_p.daemon = True
    worker_hand_p1.daemon = True
    worker_face_p1.daemon = True
    worker_handres_mailman.daemon = True
    worker_faceres_mailman.daemon = True
    worker_pref_writeman.daemon = True
    worker_svm_trainer.daemon = True

    #  worker_preprocess_p.start()
    worker_hand_p1.start()
    worker_face_p1.start()
    worker_handres_mailman.start()
    worker_faceres_mailman.start()
    worker_pref_writeman.start()
    worker_svm_trainer.start()

    with open('./profiles/profiles.pkl', 'rb') as pref_fd:
        try:
            pref_db = pkl.load(pref_fd)
        except EOFError:    # if profiles_add.pkl is empty
            pref_db = pd.DataFrame(index=[], columns=['username', 'gesture', 'location', 'scene', 'policy', 'myfeature', 'otfeature'])
            pref_db.set_index(['username'], inplace=True)

    with open('./profiles/uid2name.pkl', 'rb') as uid2name_fd:
        try:
            uid2name = pkl.load(uid2name_fd)
        except EOFError:
            uid2name = {}

    with open('./profiles/profiles_add.pkl', 'rb') as pref_add_fd:
        try:
            pref_add_db = pkl.load(pref_add_fd)
        except EOFError:    # if profiles_add.pkl is empty
            pref_add_db = pd.DataFrame(index=[], columns=['username', 'gesture', 'location', 'scene', 'policy', 'myfeature', 'otfeature'])
            pref_add_db.set_index(['username'], inplace=True)


    HOST, PORT = "", 9999
    server = Server((HOST, PORT), RequestHandler)
    ip, port = server.server_address
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        signal.pause()
    except:
        server.shutdown()
        server.server_close()
        worker_hand_p1.terminate()
        worker_face_p1.terminate()

        worker_hand_p1.join()
        worker_face_p1.join()

        dummycontinue = False
        worker_handres_mailman.join()
        worker_faceres_mailman.join()

        worker_pref_writeman.join()
        worker_svm_trainer.join()
        with open('./profiles/profiles_add.pkl', 'wb') as pref_add_fd:
            pkl.dump(pref_add_db, pref_add_fd)

