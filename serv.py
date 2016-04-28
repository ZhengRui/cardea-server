#!/usr/bin/env python

#  from gestureModel import gestDetModel

import socket
import threading
import SocketServer as SS
import signal
import struct
import numpy as np
import cv2


class RequestHandler(SS.BaseRequestHandler):
    def recvMsg(self):
        headerBytes = self.request.recv(24)
        header = struct.unpack('6i', headerBytes)
        print header

        if not header[0]:   # frame prediction message
            lat, lon = struct.unpack('<2d', self.request.recv(16))
            #  print lat, lon

            size = header[5]
            size_recv = 0
            frm_buffer = ''

            #  cv2.namedWindow("Preview", cv2.CV_WINDOW_AUTOSIZE)
            while size_recv < size:
                chunk = self.request.recv(size - size_recv)
                if not chunk:
                    print "socket receiving error"
                    return None
                else:
                    frm_buffer += chunk
                    size_recv += len(chunk)
                    #  print "size, size_recv", size, size_recv

            #  print "Frame received"

            # decode jpeg
            nparr = np.fromstring(frm_buffer, dtype=np.uint8)
            img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

            # preprocessing frame : flip , orientation
            if header[2] == 0:
                img_np = cv2.transpose(img_np)
                img_np = cv2.flip(img_np, 1-header[1])
            elif header[2] == 1:
                img_np = cv2.flip(img_np, -1)
            elif header[2] == 2:
                img_np = cv2.transpose(img_np)
                img_np = cv2.flip(img_np, header[1])
            else:
                pass

            print "Frame shape: ", img_np.shape

            #  cv2.imshow("Preview", img_np)
            #  cv2.waitKey(0)

        else:   # profile message
            pass



    def handle(self):
        print "one tcp request issued"
        msg = self.recvMsg()

class Server(SS.ThreadingMixIn, SS.TCPServer):
    pass


if __name__ == "__main__":
    #  gestureMdl = gestDetModel("/home/zerry/Work/Libs/py-faster-rcnn/models/VGG16/faster_rcnn_end2end_handGesdet/test.prototxt", "/home/zerry/Work/Libs/py-faster-rcnn/output/faster_rcnn_end2end_handGesdet/trainval/vgg16_faster_rcnn_handGesdet_aug_fulldata_iter_50000.caffemodel", 0.4, 0.8, ('natural', 'yes', 'no'))

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

