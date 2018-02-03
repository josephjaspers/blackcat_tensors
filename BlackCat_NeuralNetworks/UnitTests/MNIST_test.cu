#include "../BlackCat_NeuralNetworks.h"
#include <fstream>
#include <iostream>
#include <string>

using BC::vec;
typedef std::vector<vec> data;

namespace BC {
namespace MNIST_Test {
typedef vec tensor;

tensor expandOutput(int val, int total) {
	//Convert a value to a 1-hot output vector
	tensor out(total);
	out.zero();
	out.data()[val] = 1;
	return out;
}

tensor&  normalize(tensor& tens, double max, double min) {
	//generic feature scaling (range of [0,1])
	tens -= Scalar<double>(min);
	tens /= Scalar<double>(max - min);

	return tens;
}

void generateAndLoad(data& input_data, data& output_data, std::ifstream& read_data, int MAXVALS) {
	unsigned vals = 0;

	//load data from CSV (the first number if the correct output, the remaining 784 columns are the pixels)
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

	const int TRAINING_EXAMPLES = 2000;
	const int TRAINING_ITERATIONS = 10;

	//Generate the layers (params are: inputs, outputs)
	FeedForward f1(784, 250);
	FeedForward f2(250, 10);

	//Create the neural network
	auto network = generateNetwork(f1, f2);

	data inputs;
	data outputs;

	data testInputs;
	data testOutputs;

	//load data
	std::cout << "loading data..." << std::endl << std::endl;
	std::ifstream in_stream("///home/joseph///Downloads///train.csv");
	std::string tmp; std::getline(in_stream, tmp, '\n'); //remove headers

	//Load training examples (taken from kaggle digit recognizer train.csv)
	std::cout << " generating and loading data from csv to tensors" << std::endl;
	generateAndLoad(inputs, outputs, in_stream, TRAINING_EXAMPLES);
	in_stream.close();

	//Train
	std::cout << " training..." << std::endl;

	for (int i = 0; i < TRAINING_ITERATIONS; ++i) {
		std::cout << " iteration =  " << i << std::endl;
		for (int j = 0; j < inputs.size(); ++j) {
			vec residual = network.forwardPropagation(inputs[j]) - outputs[j];
			network.backPropagation(residual);

			//this is just the batch size
			if (j % 100 == 0) {
			network.updateWeights();
			network.clearBPStorage();
			}
		}
	}

	std::cout << "\n \n \n " << std::endl;

	for (int i = 0; i < 10; ++i) {
		std::cout << " output " << std::endl;
		outputs[i].print();
		std::cout << "prediction " << std::endl;
		network.forwardPropagation(inputs[i]).print();
		std::cout << "-----------------------------------------------------------------------------------------------------------" <<std::endl;

	}


	std::cout << " done training " << std::endl;
	return 0;
}
}
}
//int main() {
//	BC::MNIST_Test::percept_MNIST();
//	std::cout << "success" << std::endl;
//}
//
