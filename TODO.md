* Done:

    + receive/parse frame processing message
    + preprocessing (flip + rotation) of frame, leave to mobile client
    + run caffe model for each process, hand model cost 1.8G GRAM in forward(), it will be 2 hand models
      plus 2 face models running concurrently, gpu memory usage will be 97%
    + connected sockets as producer add msgs to queue for model processes to deal with
    + model processes return results from queue to producer sockets
    + add face recognition model
    + exit elegantly with ctrl+c
    + receive/parse preference updation message
    + write to preference db

* ToDo:

    + retrieve preference and action making
    + automatically retrainning svm after new features saved
    + automatically reload new svm model
    + strong / weak privacy mode
    + concurrency test (worker num, image size, task schedule)

