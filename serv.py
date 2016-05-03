#!/usr/bin/env python

from gestureModel import gestDetModel

import socket
import threading
from threading import Event
from multiprocessing import Process, JoinableQueue
from multiprocessing.dummy import Process as DummyProcess
import SocketServer as SS
import signal
import struct
import numpy as np
import cv2
import sys
import os
import time
import caffe


class RequestHandler(SS.BaseRequestHandler):

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
        #print header

        if not header[0]:   # frame prediction message
            self.res={'hand_res':None, 'hand_res_evt':Event(), 'face_res':None, 'face_res_evt':Event()}
            lat, lon = struct.unpack('<2d', self.request.recv(16))
            #  print lat, lon

            size = header[5]
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

            print "Frame received"

            # frm_raw_q.put((header[1], header[2], frm_buffer))

            # leave the efforts of rotation and flip preprocessing to the socket client,
            # no need of worker_preprocess_p and frm_raw_q, frm_buffer received here is
            # already preprocessed by the client, directly put frm_buffer to hand_inp_q
            # and face_inp_q

            hand_inp_q.put((self.name, frm_buffer))
            self.res['hand_res_evt'].wait()
            print self.res['hand_res']

        else:   # profile message
            pass

    def finish(self):
        skt_clients_map.pop(self.name)
        print "tcp request finished"
        return SS.BaseRequestHandler.finish(self)

class Server(SS.ThreadingMixIn, SS.TCPServer):
    pass

def modelPrepare(model_class_name, params, tsk_queues):
    # caffe initialization has to be put here, otherwise encounter:
    #     "syncedmem.cpp: error == cudaSuccess (3 vs. 0)"
    # or
    #     "math_functions.cu:28: CUBLAS_STATUS_SUCCESS (14 vs. 0) CUBLAS_STATUS_INTERNAL_ERROR"
    caffe.set_mode_gpu()
    caffe.set_device(0)
    modelClass = getattr(sys.modules[__name__], model_class_name)
    model = modelClass(*params)
    model.serv(*tsk_queues)


#def preProcess(raw_q, inp_qs):
    #while True:
        #front_or_back, orient_case, frm_buffer = raw_q.get()
        ## decode jpeg
        #nparr = np.fromstring(frm_buffer, dtype=np.uint8)
        #img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

        ## preprocessing frame : flip , orientation
        #if orient_case == 0:
            #img_np = cv2.transpose(img_np)
            #img_np = cv2.flip(img_np, 1-front_or_back)
        #elif orient_case == 1:
            #img_np = cv2.flip(img_np, -1)
        #elif orient_case == 2:
            #img_np = cv2.transpose(img_np)
            #img_np = cv2.flip(img_np, front_or_back)
        #else:
            #pass
        #print "Frame shape: ", img_np.shape

        ## encode back for the speed up of IPC through serialization
        #_, frm_buffer = cv2.imencode(".jpg", img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #for q in inp_qs:
            #q.put(frm_buffer)
        #raw_q.task_done()


def resMailMan(res_q, jobtype): # jobtype is 'hand_res' or 'face_res'
    while True:
        cli_name, res = res_q.get()
        client = skt_clients_map[cli_name]
        client.res[jobtype] = res
        client.res[jobtype + '_evt'].set()
        #print cli_name, " ", jobtype, " result mailed"


if __name__ == "__main__":

    import wingdbstub
    if 'WINGDB_ACTIVE' in os.environ:
        print "Success starting debug"
    else:
        print "Failed to start debug... Continuing without debug"

    skt_clients_map = {}

    #frm_raw_q = JoinableQueue()
    hand_inp_q = JoinableQueue()
    hand_res_q = JoinableQueue()

    #worker_preprocess_p = Process(target = preProcess, args = (frm_raw_q, (hand_inp_q, )))

    worker_hand_p1 = Process(target = modelPrepare, args = ('gestDetModel',
    ("/home/zerry/Work/Libs/py-faster-rcnn/models/VGG16/faster_rcnn_end2end_handGesdet/test.prototxt", "/home/zerry/Work/Libs/py-faster-rcnn/output/faster_rcnn_end2end_handGesdet/trainval/vgg16_faster_rcnn_handGesdet_aug_fulldata_iter_50000.caffemodel", 0.4, 0.8, ('natural', 'yes', 'no')), (hand_inp_q, hand_res_q)))

    worker_hand_p2 = Process(target = modelPrepare, args = ('gestDetModel', (
    "/home/zerry/Work/Libs/py-faster-rcnn/models/VGG16/faster_rcnn_end2end_handGesdet/test.prototxt","/home/zerry/Work/Libs/py-faster-rcnn/output/faster_rcnn_end2end_handGesdet/trainval/vgg16_faster_rcnn_handGesdet_aug_fulldata_iter_50000.caffemodel", 0.4, 0.8, ('natural', 'yes', 'no')), (hand_inp_q, hand_res_q)))

    worker_handres_mailman = DummyProcess(target = resMailMan, args = (hand_res_q, 'hand_res'))

    #worker_preprocess_p.daemon = True
    worker_hand_p1.daemon = True
    worker_hand_p2.daemon = True
    worker_handres_mailman.daemon = True
    #worker_preprocess_p.start()
    worker_hand_p1.start()
    worker_hand_p2.start()
    worker_handres_mailman.start()


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



