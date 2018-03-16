Last Updated: March 15, 2018
Author: Joseph Jaspers

***Stil in early stages of development
***for more fleshed out implementation (with more features)
***go to: https://github.com/josephjaspers/UNMAINTAINED-BlackCat_NeuralNetworks-Version-2/tree/master/BC_NeuralNetwork_Headers


Formatting may be obscured, go to: 
https://github.com/josephjaspers/BlackCat_Libraries/blob/master/BlackCat_NeuralNetworks3.3/README.txt

BlackCat_NeuralNetworks (BCNN) a library built on BlackCat_Tensors built for efficiency and user simplicty.
Empirically Fast (will add benchmarks soon)


Current Work:
	Add layer_type CNN
	Add layer_type Recurrent
	Add layer_type LSTM
	Add layer_type GRU

	Past implementations of CNN, RNN, LSTM, and GRU can be found here: 
	https://github.com/josephjaspers/UNMAINTAINED-BlackCat_NeuralNetworks-Version-2/tree/master/BC_NeuralNetwork_Headers
	

Installation/Setup:
	BCT is a header only library that supports compilation with the NVCC and G++ (stable with G++ 6 and 7 and NVCC CUDA 9)
	Simply add BlackCat_Tensors3.3 and BlackCat_NeuralNetworks to path and include "BlackCat_NeuralNetworks.h"

FAQ Fast Explanation:

	How to choose GPU/CPU?	Go to NN_Core/Defaults.h, change fp_tye to float, and Mathlib to GPU. (Apropriate changes already written, simply comment and uncomment the 'using' declerations fields commented)

Supports:
	GPU Multithreading (via CUDA)
	CPU Multithreading (via openMP) 

How To Use:

	Add BlackCat_Tensor3.3 to path (found: https://github.com/josephjaspers/BlackCat_Libraries)
	Add BlackCat_NeuralNetworks3.3 to path

	#include "BlackCat_NeuralNetworks.h"
	
	BC::NeuralNetwork<FeedForward, FeedForward> network(784, 250, 10); //creates a 3 layer neural network
	
	network.forwardPropagation(BC::Vector<float>); 			//forward pass through network
	network.forwardPropagation_Expression(BC::Vector<float>); 	//forward pass, do not store data for backward pass
	network.backPropagation(BC::Vector<float> output)		//backward pass, calculates error byitself
	network.updateWeights();					//updates neural network weights, gradients are stored during backProapgation.
	network.clearBCStorage();					//clear the stored gradients


	Example main (for MNIST dataset) go to:
	https://github.com/josephjaspers/BlackCat_Libraries/blob/master/BlackCat_NeuralNetworks3.3/UnitTests/MNIST_test.cpp

![BlacksTurn](https://user-images.githubusercontent.com/20384345/35011918-e1b28360-fad5-11e7-94f3-ffe79572ca1c.png)

![TestClassOutputs](https://user-images.githubusercontent.com/20384345/37546694-62f0f262-2944-11e8-99f4-ff48a92210dc.png  "TestClassOutput1")
![TestClassOutput1](https://user-images.githubusercontent.com/20384345/37546693-62e67ea4-2944-11e8-9c21-a129d2d8d94f.png  "TestClassOutput2")
![TestClassOutput2](https://user-images.githubusercontent.com/20384345/37546693-62e67ea4-2944-11e8-9c21-a129d2d8d94f.png  "TestClassOutput1")
