#include "BlackCat_NeuralNetworks.h"

#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>

namespace BC {
namespace nn {


template<class cube>
void generateAndLoad(cube& input_data, cube& output_data, std::ifstream& read_data, BC::size_t  training_examples, BC::size_t  batch_size) {

	BC::size_t  training_sets = training_examples / batch_size;

	output_data.zero();
	input_data.zero();

	std::cout << " generating loading " << std::endl;
	for (int i = 0; i < training_sets && read_data.good(); ++i) {
		for (int j = 0; j < batch_size && read_data.good(); ++j) {
			output_data[i][j].read_as_one_hot(read_data);
			input_data[i][j].read_csv_row(read_data);
		 }
	}
std::cout << " normalizing " << std::endl;

	input_data = normalize(input_data, 0, 255);
	std::cout << " post while "  << std::endl;
	std::cout << " return -- finished creating data set " << std::endl;
}
//
template<class System=BC::host_tag>
int percept_MNIST(System system_tag=System()) {

	using value_type = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube = BC::Cube<value_type, allocator_type>;
	using mat  = BC::Matrix<value_type, allocator_type>;
	using clock = std::chrono::duration<double>;





	const BC::size_t  TRAINING_EXAMPLES = 1024 * 32;
	const BC::size_t  BATCH_SIZE = 32;
	const BC::size_t  NUMB_BATCHES = TRAINING_EXAMPLES / BATCH_SIZE;
	const BC::size_t  PICTURE_SZ = 784;
	const BC::size_t  NUMB_DIGITS = 10;
	const BC::size_t  EPOCHS = 10;

    auto network = neuralnetwork(
    			feedforward(system_tag, 784, 256),
    			logistic(system_tag, 256),
    			feedforward(system_tag, 256, 10),
    			logistic(system_tag, 10),
    			outputlayer(system_tag, 10)
    );

    network.set_batch_size(BATCH_SIZE);

	cube inputs(PICTURE_SZ, BATCH_SIZE, NUMB_BATCHES);
	cube outputs(NUMB_DIGITS, BATCH_SIZE, NUMB_BATCHES);

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


	auto start = std::chrono::system_clock::now();



	for (int i = 0; i < EPOCHS; ++i) {
		std::cout << " current epoch: " << i << std::endl;
		for (int j = 0; j < NUMB_BATCHES; ++j) {
			network.forward_propagation(inputs[j]);
			network.back_propagation(outputs[j]);
			network.update_weights();
		}
	}

	auto end = std::chrono::system_clock::now();
	clock total = clock(end - start);
	std::cout << " training time: " <<  total.count() << std::endl;

	std::cout << "success " << std::endl;
	std::cout << "\n \n \n " << std::endl;


	{
	std::cout << " testing... " << std::endl;

	BC::size_t  test_images = 10;
	cube img = cube(reshape(inputs[0])(28,28, BATCH_SIZE));
	mat hyps = mat(network.forward_propagation(inputs[0]));

	for (int i = 0; i < test_images; ++i) {
		img[i].printSparse(3);
		hyps[i].print();
		std::cout << "------------------------------------" <<std::endl;
	}
	}

	std::cout << " returning " << std::endl;
	return 0;
}

}
}
