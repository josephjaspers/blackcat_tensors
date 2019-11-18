#include "../../include/BlackCat_Tensors.h"
#include "../../include/BlackCat_NeuralNetworks.h"

#include <chrono>

/*
 *
 * This tests feeds 1/4 of each MNIST image and than attempts to predict the image.
 * This proves that the recurrent neural network is functioning properly as the output is dependent
 * upon the previous blocks of the image
 *
 */
template<class System>
auto load_mnist(System system, const char* mnist_dataset, int batch_size, int samples) {

	using value_type = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube = BC::Cube<value_type, allocator_type>;

	std::ifstream read_data(mnist_dataset);
	if (!read_data.is_open()){
		std::cout << " error opening file. " << mnist_dataset << std::endl;
		throw std::invalid_argument("Failure opening mnist_train.csv");
	}

	//move the file_ptr of the csv past the headers
	std::string tmp;
	std::getline(read_data, tmp, '\n');

	int img_sz = 784; //28 * 28 gray-scale handwritten digits
	int training_sets = samples / batch_size;

	cube inputs(img_sz, batch_size, training_sets);
	cube outputs(10, batch_size, training_sets); //10 -- 1 hot vector of outputs

	//tensor's memory is uninitialized
	inputs.zero();
	outputs.zero();

	for (int i = 0; i < training_sets && read_data.good(); ++i) {
		for (int j = 0; j < batch_size && read_data.good(); ++j) {
			outputs[i][j].read_as_one_hot(read_data);
			inputs[i][j].read_csv_row(read_data);
		 }
	}

	inputs = BC::normalize(inputs, 0, 255);
	return std::make_pair(inputs, outputs);
}

template<class System=BC::host_tag>
int percept_MNIST(System system_tag, const char* mnist_dataset,
		int epochs=10, int batch_size=32, int samples=32*1024) {

	using value_type = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube = BC::Cube<value_type, allocator_type>;
	using mat  = BC::Matrix<value_type, allocator_type>;
	using vec  = BC::Vector<value_type, allocator_type>;
	using clock = std::chrono::duration<double>;

	auto network = BC::nn::neuralnetwork(
		BC::nn::lstm(system_tag, 784/4, 128),
		BC::nn::lstm(system_tag, 128, 64),
		BC::nn::feedforward(system_tag, 64, 10),
		BC::nn::softmax(system_tag, 10),
		BC::nn::logging_output_layer(system_tag, 10, BC::nn::RMSE).skip_every(100)
	);

	std::cout << "Neural Network architecture: \n" <<
			network.get_string_architecture() << std::endl;

	network.set_learning_rate(0.03);
	network.set_batch_size(batch_size);

	std::pair<cube, cube> data = load_mnist(system_tag, mnist_dataset, batch_size, samples);
	cube& inputs = data.first;
	cube& outputs = data.second;

	std::cout << " training..." << std::endl;
	auto start = std::chrono::system_clock::now();

	int img_partitions = 4;
	for (int i = 0; i < epochs; ++i){
		std::cout << " current epoch: " << i << std::endl;
		for (int j = 0; j < samples/batch_size; j++) {
			for (int p = 0; p < img_partitions; ++p) {
				auto batch = inputs[j];
				auto index = BC::index(0,784 * (p/(float)img_partitions));
				auto shape = BC::shape(784/4, batch_size);

				network.forward_propagation(batch[{index, shape}]);
			}
				//Apply backprop on the last two images (images 3/4 and 4/4)
				network.back_propagation(outputs[j]);
				network.back_propagation(outputs[j]);

				network.update_weights();
		}
	}

	auto end = std::chrono::system_clock::now();
	clock total = clock(end - start);
	std::cout << " training time: " <<  total.count() << std::endl;

	std::cout << " testing... " << std::endl;

	network.copy_training_data_to_single_predict(0);
	{
		auto batch = inputs[0];
		auto shape = BC::shape(784/4, batch_size);
		for (int p = 0; p < img_partitions-1; ++p) {
			auto index = BC::index(0,784 * (p/(float)img_partitions));
			network.predict(batch[{index, shape}]);
		}

		auto last_index = BC::index(0,784 * ((img_partitions-1)/(float)img_partitions));
		mat hyps =network.predict(batch[{last_index, shape}]);

		BC::size_t test_images = 10;
		cube img = cube(reshape(inputs[0], BC::shape(28,28, batch_size)));
		for (int i = 0; i < test_images; ++i) {
			img[i].t().print_sparse(3);
			hyps[i].print();
			std::cout << "------------------------------------" <<std::endl;
		}
	}


	{

		BC::size_t test_images = 10;
		cube img = cube(reshape(inputs[0], BC::shape(28,28, batch_size)));
		for (int i = 0; i < test_images; ++i) {

			auto batch = inputs[0];
			auto shape = BC::shape(784/4, batch_size);
			for (int p = 0; p < img_partitions-1; ++p) {
				auto index = BC::index(0,784 * (p/(float)img_partitions));
				network.single_predict(batch[{index, shape}][i]);
			}
			auto last_index = BC::index(0,784 * ((img_partitions-1)/(float)img_partitions));
			vec hyps = network.single_predict(batch[{last_index, shape}][i]);


			img[i].t().print_sparse(3);
			hyps.print();
			std::cout << "------------------------------------" <<std::endl;
		}
	}

	std::cout << " success " << std::endl;
	return 0;
}
