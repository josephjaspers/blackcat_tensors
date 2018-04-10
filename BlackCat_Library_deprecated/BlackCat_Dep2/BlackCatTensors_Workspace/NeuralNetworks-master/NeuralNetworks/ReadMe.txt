========================================================================
    CONSOLE APPLICATION : NeuralNetworks Project Overview
========================================================================
A Neural Network library supporting written for the CPU utilizing OpenMP (2.0) for multithreading optimization support.

The library contains implementations of:

Generic Recurrent Layer
Gated Recurrent Unit Layer
Long Short-term Memory Layer
FeedForward Layer
FeedForward Layer (no recurrence support) 

The library is used in conjunction with the static library jas_Matrices (included in my Github). 
Check the mainTests file to see examples of the NN.


Currently being developed:

Convolutional NeuralNetwork Layer (was once supported but since the Matrix-class switch it has become deprecated)
Matrix Class -Cuda (for GPU optimization) 
