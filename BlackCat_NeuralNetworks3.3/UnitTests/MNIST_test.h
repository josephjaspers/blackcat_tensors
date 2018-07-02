//#include "../BlackCat_GPU_NeuralNetworks.h"
#include "../BlackCat_NeuralNetworks.h"

#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
using BC::NN::vec;
using BC::NN::scal;
using BC::NN::mat;
typedef std::vector<vec> data;
typedef vec tensor;

namespace BC {
namespace NN {
namespace MNIST_Test {

//int correct(const vec& hypothesis, const vec& output) {
//	int h_id = 0;
//	int o_id = 0;
//
//	double h_max = hypothesis(0);
//	double o_max = output(0);
//
//	for (int i = 1; i < hypothesis.size(); ++i) {
//		if (h_max < hypothesis(i)) {
//			h_max = hypothesis(i);
//			h_id = i;
//		}
//		if (o_max < output(i)) {
//			o_max = output(i);
//			o_id = i;
//		}
//	}
//	return h_id == o_id;
//}

void generateAndLoad(cube& input_data, cube& output_data, std::ifstream& read_data, int training_examples, int batch_size) {

	int training_sets = training_examples / batch_size;

	std::cout << " generating loading " << std::endl;
	for (int i = 0; i < training_sets && read_data.good(); ++i) {
		for (int j = 0; j < batch_size && read_data.good(); ++j) {
			vec out(10);
			out.read_as_one_hot(read_data);
			output_data[i][j] = out;

			vec in(784);
			in.read(read_data, false);
			input_data[i][j] = in;
		}
	}

	input_data = normalize(input_data, 0, 255);
	std::cout << " post while "  << std::endl;
	std::cout << " return -- finished creating data set " << std::endl;
}

int percept_MNIST() {

	const int TRAINING_EXAMPLES =  2000;
	const int BATCH_SIZE = 100;
	const int NUMB_BATCHES = TRAINING_EXAMPLES / BATCH_SIZE;

	const int EPOCHS = 100;

	NeuralNetwork<FeedForward> network(784, 10);
	network.setLearningRate(.03);
	network.set_batch_size(BATCH_SIZE);

	cube inputs(784, BATCH_SIZE, NUMB_BATCHES);
	cube outputs(10, BATCH_SIZE, NUMB_BATCHES);

	//load data
	std::cout << "loading data..." << std::endl << std::endl;
	std::ifstream in_stream("///home/joseph///Downloads///train.csv");
	std::string tmp; std::getline(in_stream, tmp, '\n');  	//remove headers

	//Load training examples (taken from kaggle digit recognizer train.csv)
	std::cout << " generating and loading data from csv to tensors" << std::endl;
	generateAndLoad(inputs, outputs, in_stream, TRAINING_EXAMPLES, BATCH_SIZE);
	std::cout << " post load "  << std::endl;

	in_stream.close();

	//Train
	std::cout << " training..." << std::endl;
	float t = omp_get_wtime();

	for (int i = 0; i < EPOCHS; ++i) {
		for (int j = 0; j < NUMB_BATCHES; ++j) {
			network.forward_propagation(inputs[j]);

			network.back_propagation(outputs[j]);
			network.update_weights();
			network.clear_stored_delta_gradients();
		}
	}
//
//
//
	t = omp_get_wtime() - t;
	printf("It took me %f clicks (%f seconds).\n", t, ((float) t));
	std::cout << "success " << std::endl;
//
//	float correct_ = 0;
//	for (int i = 0; i < inputs.size(); ++i) {
//		if (correct(network.forward_propagation(inputs[i]), outputs[i])) {
//			++correct_;
//		}
//	}
//	std::cout << " correct: " << correct_/inputs.size()  <<std::endl;

	std::cout << "\n \n \n " << std::endl;
	std::cout << " testing... " << std::endl;

	int test_images = 10;

	mat img_adj(28,28);
//
	cube img = cube(reshape(inputs[0])(28,28, BATCH_SIZE));
	mat hyps = mat(network.forward_propagation(inputs[0]));
	for (int i = 0; i < test_images; ++i) {
		img_adj = img[i].t();
		img_adj.printSparse(3);
		hyps[i].print();
		std::cout << "-----------------------------------------------------------------------------------------------------------" <<std::endl;
	}
	return 0;
}

}
}
}
