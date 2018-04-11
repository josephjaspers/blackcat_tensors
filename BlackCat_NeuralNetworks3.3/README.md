Last Updated: March 16, 2018
Author: Joseph Jaspers

BlackCat_NeuralNetworks (BCNN) a library built on BlackCat_Tensors built for efficiency and user simplicty.


Mini-update April 10th 2018:

	BlackCat_NeuralNetworks is now a thread-safe library. Designed to support batch-multithreading with openMP
	The NeuralNetwork object methods forwardPropagation(), and backPropagation() are now thread-safe
	updateWeights() and clearBPStorage() ARE NOT thread-safe, they are designed to update the NeuralNetwork after each batch
	
	Future Work: Recurrent/Gru/LSTM will be added within a week, Convolution will hopefully be added within 2 weeks
	

Past implementations of CNN, RNN, LSTM, and GRU can be found here:
	https://github.com/josephjaspers/UNMAINTAINED-BlackCat_NeuralNetworks-Version-2/tree/master/BC_NeuralNetwork_Headers
	

Installation/Setup:

	BCT is a header only library that supports compilation with the NVCC and G++ 
	(stable with G++ 6 and 7 and NVCC CUDA 9)
	Add BlackCat_Tensors3.3 and BlackCat_NeuralNetworks to path and include "BlackCat_NeuralNetworks.h"

FAQ Fast Explanation:

	How to choose GPU/CPU?	
	-Go to NN_Core/Defaults.h
	- using ml = CPU;             // Change to using ml = GPU;
	- using fp_type = double;     // Change to using fp_type = float;
	
Supports:

	GPU Multithreading (via CUDA)
	CPU Multithreading (via openMP) 

How To Use:

	//Add BlackCat_Tensor3.3 to path (found: https://github.com/josephjaspers/BlackCat_Libraries)
	//Add BlackCat_NeuralNetworks3.3 to path

	#include "BlackCat_NeuralNetworks.h"
	
	BC::NeuralNetwork<FeedForward, FeedForward> network(784, 250, 10); //creates a 3 layer neural network
	
	network.forwardPropagation(BC::Vector<float>); 			//forward pass through network
	network.forwardPropagation_Expression(BC::Vector<float>); 	//forward pass, do not store data for backward pass
	network.backPropagation(BC::Vector<float>)			//backward pass, calculates error byitself
	network.updateWeights();					//updates neural network weights, gradients are stored during backProapgation.
	network.clearBPStorage();					//clear the stored gradients


Example main (for MNIST dataset) go to:
	https://github.com/josephjaspers/BlackCat_Libraries/blob/master/BlackCat_NeuralNetworks3.3/UnitTests/MNIST_test.cpp


TestClass outputs 

![TestClassOutputs](https://user-images.githubusercontent.com/20384345/37546694-62f0f262-2944-11e8-99f4-ff48a92210dc.png  "TestClassOutput1")
![TestClassOutput1](https://user-images.githubusercontent.com/20384345/37546692-62dce43e-2944-11e8-9d3d-236ee151ebfa.png  "TestClassOutput2")
![TestClassOutput2](https://user-images.githubusercontent.com/20384345/37546693-62e67ea4-2944-11e8-9c21-a129d2d8d94f.png  "TestClassOutput1")
