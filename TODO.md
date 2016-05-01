* Done:

    + receive/parse frame processing message
    + preprocessing (flip + rotation) of frame, leave to mobile client
    + run caffe model for each process, hand model cost 1.8G GRAM in forward(), it will be 2 hand models
      plus 2 face models running concurrently
    + connected sockets as producer add msgs to queue for model processes to deal with

* ToDo:

    + model processes return results from queue to producer sockets
    + add face recognition model
    + retrieve preference and action making
    + receive/parse preference updation message
    + write to preference db
    + automatically retrainning svm after new features saved
    + automatically reload new svm model
    + strong / weak privacy mode
    + concurrency test (worker num, image size, task schedule)

