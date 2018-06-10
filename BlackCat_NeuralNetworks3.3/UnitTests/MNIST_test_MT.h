/*
 * MNIST_test_MT.cpp
 *
 *  Created on: Mar 24, 2018
 *      Author: joseph
 */

#ifndef MNIST_TEST_MT_CPP_
#define MNIST_TEST_MT_CPP_

#include "../BlackCat_NeuralNetworks.h"
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <pthread.h>


using BC::NN::vec;
using BC::NN::scal;
using BC::NN::mat;
typedef std::vector<vec> data;
typedef vec tensor;
#include <iostream>


#include <cxxabi.h>

std::string removeNS( const std::string & source, const std::string & namespace_ )
{
    std::string dst = source;
    size_t position = source.find( namespace_ );
    while ( position != std::string::npos )
    {
        dst.erase( position, namespace_.length() );
        position = dst.find( namespace_, position + 1 );
    }
    return dst;
}

template<class T>
std::string type_name() {
	int status;
	  std::string demangled = abi::__cxa_demangle(typeid(T).name(),0,0,&status);
	  return removeNS(removeNS(removeNS(demangled, "BC::"), "internal::"), "function::");
}


namespace BC {
namespace NN {
namespace MNIST_Test_MT {

tensor expandOutput(int val, int total) {
	//Convert a value to a 1-hot output vector
	tensor out(total);
	out.zero();
	out(val) = 1;
	return out;
}

tensor&  normalize(tensor& tens, fp_type max, fp_type min) {
	//generic feature scaling (range of [0,1])
	tens -= scal(min);
	tens /= scal(max - min);

	return tens;
}
bool correct(const vec& hypothesis, const vec& output) {
	int h_id = 0;
	int o_id = 0;

	double h_max = hypothesis.internal()[0];
	double o_max = output.internal()[0];

	for (int i = 1; i < hypothesis.size(); ++i) {
		if (hypothesis.internal()[i] > h_max) {
			h_max = hypothesis.internal()[i];
			h_id = i;
		}
		if (output.internal()[i] > o_max) {
			o_max = output.internal()[i];
			o_id = i;
		}
	}
	return h_id == o_id;
}



void generateAndLoad(data& input_data, data& output_data, std::ifstream& read_data, int MAXVALS) {
	unsigned vals = 0;

	//load data from CSV (the first number if the correct output, the remaining 784 columns are the pixels)
	std::cout << " generating loading " << std::endl;
	while (read_data.good() && vals < MAXVALS) {

		std::string output;
		std::getline(read_data, output, ',');
		int out = std::stoi(output);

		tensor input(784);

		input.read(read_data, false);
		output_data.push_back(expandOutput(out, 10));
		normalize(input, 255, 0);

		input_data.push_back(input);

		++vals;
	}
	std::cout << " return -- finished creating data set " << std::endl;
}

int percept_MNIST() {

	const int TRAINING_EXAMPLES =  2000;
	const int TRAINING_ITERATIONS = 10;
	const int NUMB_THREADS = 4;
	const int BATCH_SIZE = 8;
	//Create the neural network
	NeuralNetwork<FeedForward, FeedForward> network(784, 250, 10);
	network.setLearningRate(.03);
	network.set_omp_threads(NUMB_THREADS);

	omp_set_num_threads(NUMB_THREADS);


	/*
	 * 	MULTITHREADED NEURAL NETWORK IS STILL IN BETA
	 *	BATCH_SIZE MUST EQUAL THREAD NUMBERS, ---->
	 *	THIS WILL BE CHANGED IN FUTURE
	 */

	data inputs;
	data outputs;

	data testInputs;
	data testOutputs;

	//load data
	std::cout << "loading data..." << std::endl << std::endl;
	std::ifstream in_stream("///home/joseph///Downloads///train.csv");
	std::string tmp; std::getline(in_stream, tmp, '\n');  	//remove headers

	//Load training examples (taken from kaggle digit recognizer train.csv)
	std::cout << " generating and loading data from csv to tensors" << std::endl;
	generateAndLoad(inputs, outputs, in_stream, TRAINING_EXAMPLES);
	in_stream.close();

	//Train
	float t;
	t = omp_get_wtime();
	printf("\n Calculating... BC_NN training time \n");

	std::cout << " training..." << std::endl;

	for (int i = 0; i < TRAINING_ITERATIONS; ++i) {
		for (int j = 0; j < inputs.size(); j += BATCH_SIZE) {
			int numb_instances = j + BATCH_SIZE < inputs.size() ? j + BATCH_SIZE : inputs.size();
#pragma omp parallel for schedule(static)
			for (int k = j; k < numb_instances; ++k) {
				network.forwardPropagation(inputs[j]);
				network.backPropagation(outputs[j]);
			}
#pragma omp barrier
			network.updateWeights();
			network.clearBPStorage();
		}
	}


	t = omp_get_wtime() - t;
	printf("It took me %f clicks (%f seconds).\n", t, ((float) t));
	std::cout << "success " << std::endl;

	float correct_ = 0;
	for (int i = 0; i < inputs.size(); ++i) {
		if (correct(network.forwardPropagation(inputs[i]), outputs[i])) {
			++correct_;
		}
	}
	std::cout << " correct: " << correct_/inputs.size()  <<std::endl;

	std::cout << "\n \n \n " << std::endl;
	std::cout << " testing... " << std::endl;

	mat img(28,28);
	mat img_adj(28,28);

	for (int i = 0; i < 10; ++i) {
		std::cout << " output " << std::endl;
		img = inputs[i];//outputs[i];//.print();
		img_adj = img.t();
		img_adj.printSparse(3);

		std::cout << "prediction " << std::endl;
		network.forwardPropagation(inputs[i]).print();
		std::cout << "-----------------------------------------------------------------------------------------------------------" <<std::endl;
	}
	return 0;
}

}
}
}


#endif /* MNIST_TEST_MT_CPP_ */
