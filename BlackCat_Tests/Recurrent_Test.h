//#include "../BlackCat_GPU_NeuralNetworks.h"

#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>

namespace BC {
namespace NN {
namespace rec_MNIST_test {
using BC::NN::vec;
using BC::NN::scal;
using BC::NN::mat;
using BC::NN::cube;
using tensor4d = BC::Tensor<4, double, BC::CPU>;


void generateAndLoad(cube& input_data, cube& output_data, std::ifstream& read_data, int training_examples, int batch_size) {

	int training_sets = training_examples / batch_size;

	output_data.zero();
	input_data.zero();

	std::cout << " generating loading " << std::endl;
	for (int i = 0; i < training_sets && read_data.good(); ++i) {
		for (int j = 0; j < batch_size && read_data.good(); ++j) {
			output_data[i][j].read_as_one_hot(read_data);
			input_data[i][j].read(read_data);
		}
	}

	input_data = normalize(input_data, 0, 255);
	std::cout << " post while "  << std::endl;
	std::cout << " return -- finished creating data set " << std::endl;
}

int percept_MNIST() {
//
	const int TRAINING_EXAMPLES =  1024 * 4;
	const int BATCH_SIZE = 128;
	const int NUMB_BATCHES = TRAINING_EXAMPLES / BATCH_SIZE;
	const int PICTURE_SZ = 784;
	const int NUMB_DIGITS = 10;
	const int EPOCHS = 10;

	NeuralNetwork<Recurrent_FeedForward, Rec> network(28, 128, 10);
	network.set_learning_rate(.03);
	network.set_max_bptt_length(28);
	network.set_batch_size(BATCH_SIZE);
//	network.initialize_variables();

	cube inputs(PICTURE_SZ, BATCH_SIZE, NUMB_BATCHES);
	cube outputs(NUMB_DIGITS, BATCH_SIZE, NUMB_BATCHES);

	BC::Tensor_Shared<4, double> recurrent_inputs(28, 28, BATCH_SIZE, NUMB_BATCHES);
	recurrent_inputs.internal() = inputs.internal();
	recurrent_inputs.print_dimensions();
	auto inputs_fixed = format_as(recurrent_inputs, 0, 1, 3, 2);
	inputs_fixed.print_dimensions();
	inputs_fixed.print_leading_dimensions();
	std::cout << "loading data..." << std::endl << std::endl;
	std::ifstream in_stream("///home/joseph///Datasets///all///train.csv");

	if (!in_stream.is_open()){
		std::cout << " error opening file" << std::endl;
		return -1;
	}

	//move the file_ptr of the csv past the headers
	std::string tmp; std::getline(in_stream, tmp, '\n');

	//Load training examples (taken from kaggle digit recognizer train.csv)
	std::cout << " generating and loading data from csv to tensors" << std::endl;
	generateAndLoad(inputs, outputs, in_stream, TRAINING_EXAMPLES, BATCH_SIZE);
	std::cout << " post load "  << std::endl;
	in_stream.close();

	std::cout << " training..." << std::endl;
//	float t = omp_get_wtime();

	inputs_fixed.print_dimensions();
	for (int i = 0; i < EPOCHS; ++i) {
		std::cout << " current epoch: " << i << std::endl;
		for (int j = 0; j < NUMB_BATCHES; ++j) {
			for (int r= 0; r< 28; ++r)  {
			network.forward_propagation(inputs_fixed[j][r]);
			}
			for (int r= 0; r< 28; ++r)  {
			network.back_propagation(outputs[j]);
			network.cache_gradients();
			}

			network.update_weights();
//			network.clear_stored_delta_gradients();
		}
	}
//	t = omp_get_wtime() - t;


//	printf("It took me %f clicks (%f seconds).\n", t, ((float) t));
	std::cout << "success " << std::endl;

	std::cout << "\n \n \n " << std::endl;
	std::cout << " testing... " << std::endl;

	int test_images = 10;
	mat img_adj(28,28);

	cube img = cube(reshape(inputs[0])(28,28, BATCH_SIZE));
	mat hyps = mat(network.forward_propagation(inputs[0]));
	for (int i = 0; i < test_images; ++i) {
		img_adj = img[i].t();
		img_adj.printSparse(3);

		for (int r = 0; r < 28; ++r) {
			hyps[i] = network.forward_propagation(reshape(inputs[i][r])(28,1));
			hyps[i].print();
		}

		std::cout << "-----------------------------------------------------------------------------------------------------------" <<std::endl;
	}

	std::cout << " returning " << std::endl;
	return 0;
}

}
}
}
