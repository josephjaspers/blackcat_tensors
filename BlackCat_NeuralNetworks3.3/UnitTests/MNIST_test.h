#include "../BlackCat_NeuralNetworks.h"
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
using BC::NN::vec;
using BC::NN::scal;
using BC::NN::mat;
typedef std::vector<vec> internal;
typedef vec tensor;

namespace BC {
namespace NN {
namespace MNIST_Test {

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

	double h_max = hypothesis(0);
	double o_max = output(0);

	for (int i = 1; i < hypothesis.size(); ++i) {
		if (h_max < hypothesis(i)) {
			h_max = hypothesis(i);
			h_id = i;
		}
		if (o_max < output(i)) {
			o_max = output(i);
			o_id = i;
		}
	}
	return h_id == o_id;
}



void generateAndLoad(internal& input_internal, internal& output_internal, std::ifstream& read_internal, int MAXVALS) {
	unsigned vals = 0;

	//load internal from CSV (the first number if the correct output, the remaining 784 columns are the pixels)
	std::cout << " generating loading " << std::endl;
	while (read_internal.good() && vals < MAXVALS) {

		std::string output;
		std::getline(read_internal, output, ',');
		int out = std::stoi(output);

		tensor input(784);
		input.read(read_internal, false);
		output_internal.push_back(expandOutput(out, 10));
		normalize(input, 255, 0);

		input_internal.push_back(input);

		++vals;
	}
	std::cout << " return -- finished creating internal set " << std::endl;
}

int percept_MNIST() {

	const int TRAINING_EXAMPLES =  2000;
	const int TRAINING_ITERATIONS = 100;

//	Generate the layers (params are: inputs, outputs)
	//Create the neural network
	NeuralNetwork<FeedForward> network(784, 10);
//	NeuralNetwork<FeedForward, Recurrent> network(784, 250, 10);
	network.setLearningRate(.03);

	internal inputs;
	internal outputs;
	internal testInputs;
	internal testOutputs;


	//load internal
	std::cout << "loading internal..." << std::endl << std::endl;
	std::ifstream in_stream("///home/joseph///Downloads///train.csv");
	std::string tmp; std::getline(in_stream, tmp, '\n');  	//remove headers

	//Load training examples (taken from kaggle digit recognizer train.csv)
	std::cout << " generating and loading internal from csv to tensors" << std::endl;
	generateAndLoad(inputs, outputs, in_stream, TRAINING_EXAMPLES);
	in_stream.close();


	//Train
	float t;
	t = omp_get_wtime();
	printf("\n Calculating... BC_NN training time \n");

	std::cout << " training..." << std::endl;

	for (int i = 0; i < TRAINING_ITERATIONS; ++i) {
		std::cout << " iteration =  " << i << std::endl;
		for (int j = 0; j < inputs.size(); ++j) {
			network.forwardPropagation(inputs[j]);
			network.backPropagation(outputs[j]);

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
