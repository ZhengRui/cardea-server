import cv2
import caffe
import dlib
import numpy as np
import Queue

from align.align_dlib import AlignDlib as ADlib
from svmutil import *

# Markers 36, 45, 48, 54 's target coordinates in a 128x128 image
Anchors = np.float32([
    (29.08151817, 40.56148148), (90.19503021, 39.43851852),
    (40.68895721, 88.38999939), (80.59230804, 87.61001587)
    ])

lower = np.array([0, 48, 60], dtype="uint8")
upper = np.array([30, 255, 255], dtype="uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

class faceRecModel:
    def __init__(self, cascade, mdl_landmarks, proto_caffe, mdl_caffe, blob_name, mdl_start_pth):
        self.cascade = cv2.CascadeClassifier(cascade)
        self.alignment = ADlib(mdl_landmarks)
        self.net = caffe.Net(proto_caffe, mdl_caffe, caffe.TEST)
        self.blob_name = blob_name
        self.classify = svm_load_model(mdl_start_pth)

    def faceDetCV(self, frm, skinFilter=True):
        bbs = self.cascade.detectMultiScale(cv2.equalizeHist(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)), scaleFactor=1.1, minNeighbors=3, minSize=(40, 40), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        bbs = bbs if len(bbs) else np.array([])
        if len(bbs) and skinFilter:
            keep = np.array([False] * bbs.shape[0])
            i = 0
            for (x,y,w,h) in bbs:
                bb_hsv = cv2.cvtColor(frm[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
                skinMask = cv2.inRange(bb_hsv, lower, upper)
                skinMask = cv2.dilate(skinMask, kernel, iterations=2)
                skinMask = skinMask[h/5 : h, w/4 : 3*w/4]
                if np.sum(skinMask) * 2 / (h * w * 255 ) > 0.4:
                    keep[i] = True
                i += 1
            return bbs[keep]
        else:
            return bbs

    def faceDetAlign(self, frm):
        bbs = self.faceDetCV(frm, skinFilter=True)
        print "detected ", len(bbs), " faces"

        faces = None

        scale = max(frm.shape[:2]) / 960.0
        frm_small = cv2.resize(frm, (int(frm.shape[1] / scale), int(frm.shape[0] / scale)), interpolation=cv2.INTER_AREA)
        frm_small = cv2.cvtColor(frm_small, cv2.COLOR_BGR2RGB)
        bbs_small = (bbs / scale).astype('int')
        xmax = frm_small.shape[1]
        ymax = frm_small.shape[0]

        bbs_filt = []
        offset = 15

        for (x,y,w,h) in bbs_small:
            x0 = max(x-offset, 0)
            y0 = max(y-offset, 0)
            x1 = min(x+w+2*offset, xmax-1)
            y1 = min(y+h+2*offset, ymax-1)
            bb = self.alignment.getLargestFaceBoundingBox(np.array(frm_small[y0:y1, x0:x1]))
            if bb is None:
                print 'CV false positive'
                continue
            bb = dlib.rectangle(int((bb.left()+x0)*scale), int((bb.top()+y0)*scale), int((bb.right()+x0)*scale), int((bb.bottom()+y0)*scale))

            bbs_filt.append(bb)
            landmarks = np.float32(self.alignment.findLandmarks(frm, bb))
            H = cv2.getAffineTransform(np.vstack((landmarks[[36, 45]], (landmarks[48]+landmarks[54])/2.)), np.vstack((Anchors[[0, 1]], (Anchors[2]+Anchors[3])/2.)))
            face_aligned = cv2.warpAffine(frm, H, (128, 128))
            face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_RGB2GRAY)
            face_aligned = cv2.equalizeHist(face_aligned)
            face_aligned = face_aligned[np.newaxis, np.newaxis, :, :] / 255.0
            faces = face_aligned if faces is None else np.vstack((faces, face_aligned))
        return faces, bbs_filt

    def featureExtract(self, faces):
        if faces is None:
            return []

        self.net.blobs['data'].reshape(*faces.shape)
        self.net.blobs['data'].data[...] = faces
        self.net.forward()
        feats = self.net.blobs[self.blob_name].data
        return feats

    def runSVM(self, feature, model):
        p_label, p_acc, p_vals = svm_predict([0] * len(feature), feature, model, '-b 1 -q')
        return p_label, p_vals

    def serv(self, inp_q, res_q, continue_evt, reload_val, mdl_update_pth):
        while True:
            try:
                continue_evt.wait()
                if reload_val.value:
                    self.classify = svm_load_model(mdl_update_pth)
                    reload_val.value = 0

                cli_name, data_buffer, mode = inp_q.get(timeout=2)
                bbs_filt, labels, probs = [], [], []

                if mode:    # 1 is strong mode
                    bbs_filt_np, feats = data_buffer
                    bbs_filt_np = bbs_filt_np.astype(np.int64)
                    for bbs_np in bbs_filt_np:
                        bbs_filt.append(dlib.rectangle(bbs_np[0], bbs_np[1], bbs_np[2], bbs_np[3]))

                else:
                    nparr = np.fromstring(data_buffer, dtype=np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

                    faces, bbs_filt = self.faceDetAlign(img_np)
                    feats = self.featureExtract(faces)

                if len(feats):

                    labels, probs = self.runSVM(feats.tolist(), self.classify)

                res_q.put((cli_name, (bbs_filt, labels, probs, feats)))
                inp_q.task_done()
            except (Queue.Empty, KeyboardInterrupt):
                #  print 'timeout in face'
                pass
