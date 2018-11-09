Last Updated: June 25th, 2018
Author: Joseph Jaspers

BlackCat_NeuralNetworks (BCNN) is a library built on BlackCat_Tensors built for efficiency and user simplicty.


    BlackCat_NeuralNetworks semi-thread-safe library. Designed to support batch-multithreading with openMP
    The NeuralNetwork object methods forward_propagation(), and back_propagation() are now thread-safe
    update_weights() and clear_stored_delta_gradients() ARE NOT thread-safe, they are designed to update the NeuralNetwork after each batch 
    (check the UnitTests folder for examples)
    
    ***Currently Working on adding ConvNets and LSTM***

Past implementations of CNN, RNN, LSTM, and GRU can be found here:
    https://github.com/josephjaspers/UNMAINTAINED-BlackCat_NeuralNetworks-Version-2/tree/master/BC_NeuralNetwork_Headers

Installation/Setup:

    BCT is a header only library that supports compilation with the NVCC and G++ 
    (stable with G++ 6 and 7 and NVCC CUDA 9)
    Add BlackCat_Tensors3.3 and BlackCat_NeuralNetworks to path and include "BlackCat_NeuralNetworks.h" (for CPU) and "BlackCat_GPU_NeuralNetowrks.h (for CUDA)

FAQ Fast Explanation:

    How to choose GPU/CPU?    
    #include "BlackCat_NeuralNetworks.h"        //CPU implementation
    #include "BlackCat_GPU_NeuralNetworks.h"     //CUDA implementation 
    
Supports:

    GPU Multithreading (via CUDA)
    CPU Multithreading (via openMP) 

How To Use:
    
    BC::NeuralNetwork<FeedForward, FeedForward> network(784, 250, 10); //creates a 3 layer neural network
    
    network.forward_propagation(BC::Vector<float>);             //forward pass through network
    network.forward_propagation_expression(BC::Vector<float>);     //forward pass, do not store data for backward pass (optimized for post-training)
    network.back_propagation(BC::Vector<float>)            //backward pass, calculates automatically calculates residual
    network.update_weights();                    //updates neural network weights, based on gradients are stored during backProapgation.
    network.clear_stored_delta_gradients();                    //clear the stored gradients


Example main (for MNIST dataset):
    https://github.com/josephjaspers/BlackCat_Libraries/blob/master/BlackCat_NeuralNetworks3.3/UnitTests/MNIST_test.cpp


TestClass outputs 

![TestClassOutputs](https://user-images.githubusercontent.com/20384345/37546694-62f0f262-2944-11e8-99f4-ff48a92210dc.png  "TestClassOutput1")
![TestClassOutput1](https://user-images.githubusercontent.com/20384345/37546692-62dce43e-2944-11e8-9d3d-236ee151ebfa.png  "TestClassOutput2")
![TestClassOutput2](https://user-images.githubusercontent.com/20384345/37546693-62e67ea4-2944-11e8-9c21-a129d2d8d94f.png  "TestClassOutput1")
