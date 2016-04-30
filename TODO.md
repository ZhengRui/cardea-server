* Done:

    + receive/parse frame processing message
    + preprocessing (flip + rotation) of frame
    + run caffe model for each process, hand model cost 1.8G GRAM in forward(), it will be 2 hand models
      plus 2 face models running concurrently
    + connected sockets as producer add msgs to queue for model processes to deal with

* ToDo:

    + model processes return results from queue to producer sockets
    + receive/parse preference updation message
    + write to preference db

