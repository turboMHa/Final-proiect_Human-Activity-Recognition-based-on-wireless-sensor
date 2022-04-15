1. Environments
    Python: 3.7.0
    Matlab: R2021b
    Tensorflow:1.15.0
     
2. Files Description
    data: Traning data for neural network
    Real Time Data: data collected by phone
    model: the model saving in this files after traning
    lstm_architecture.py: Lstm architecture
    main.py: main function
    matlabTX.m: collect the sensor data from phone
    savedModel1.py: model after traing and it can classify the real time activities
    sliding_window.py: sliding window for processing collected data

3. How to use
    Through the main.py file hyperparameters, can adjust the structure of the neural network
    use_bidirectionnal_cells = False  n_stacked_layers = 0  Basic LSTM 
    use_bidirectionnal_cells = True   n_stacked_layers = 0  Bidirectional LSTM
    use_bidirectionnal_cells = False  n_stacked_layers = 0  Resnet LSTM 
    use_bidirectionnal_cells = False  n_stacked_layers = 0  Res-Bidir LSTM  