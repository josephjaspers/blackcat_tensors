#include "../../blackcat/tensors.h"
#include "../../blackcat/neural_networks.h"
#include "../datasets/mnist_loader.h"
#include <chrono>
#include <string>

template<class System=bc::host_tag>
int percept_MNIST(System system_tag, std::string mnist_dataset,
		int epochs=10, int batch_size=32, int samples=32*1024) {

	using value_type     = typename System::default_floating_point_type;
	using allocator_type = bc::Allocator<value_type, System>;
	using cube           = bc::Cube<value_type, allocator_type>;
	using mat            = bc::Matrix<value_type, allocator_type>;
	using clock          = std::chrono::duration<double>;

	auto network = bc::nn::neuralnetwork(
		bc::nn::lstm(system_tag, 784/4, 128),
		bc::nn::lstm(system_tag, 128, 64),
		bc::nn::feedforward(system_tag, 64, 10),
		bc::nn::softmax(system_tag, 10),
		bc::nn::logging_output_layer(system_tag, 10, bc::nn::RMSE).skip_every(100)
	);


	//Regression-test to ensure compilation 
	if (false) {
		auto lstm = bc::nn::lstm(system_tag, 784 / 4, 128);
		lstm = lstm;
		auto ff = bc::nn::feedforward(system_tag, 64, 10);
		ff = ff;
		auto softmax = bc::nn::softmax(system_tag, 10);
		softmax = softmax;
		network = network;
	}

	bc::print("Neural Network architecture:");
	bc::print(network.get_string_architecture());

	network.set_learning_rate(.003);
	network.set_batch_size(batch_size);

	std::pair<cube, cube> data = load_mnist(
			system_tag, mnist_dataset, batch_size, samples);

	cube& inputs = data.first;
	cube& outputs = data.second;

	bc::print("training...");
	auto start = std::chrono::system_clock::now();

	int img_partitions = 4;
	for (int i = 0; i < epochs; ++i){
		bc::print("current epoch:  ", i);

		for (int j = 0; j < samples/batch_size; j++) {

			for (int p = 0; p < img_partitions; ++p) {
				auto batch = inputs[j];
				auto index = bc::dim(0,784 * (p/(float)img_partitions));
				auto shape = bc::dim(784/4, batch_size);
				network.forward_propagation(batch[{index, shape}]);
			}

			//Apply backprop on the last two images (images 3/4 and 4/4)
			network.back_propagation(outputs[j]);
			network.back_propagation(outputs[j]);

			network.update_weights();
		}
	}

	auto end = std::chrono::system_clock::now();
	bc::print("training time:", clock(end - start).count());

	bc::print("testing...");
	network.copy_training_data_to_single_predict(0);

	auto batch = inputs[0];
	auto shape = bc::dim(784/4, batch_size);

	for (int p = 0; p < img_partitions-1; ++p) {
		auto index = bc::dim(0, 784*(p/(float)img_partitions));
		network.predict(batch[{index, shape}]);
	}

	auto last_index = bc::dim(0,784*((img_partitions-1)/(float)img_partitions));
	mat hyps =network.predict(batch[{last_index, shape}]);

	bc::size_t test_images = 10;
	cube img = cube(inputs[0].reshaped(28,28, batch_size));
	for (int i = 0; i < test_images; ++i) {
		mat(bc::logical(img[i].t())).print_sparse(0);
		hyps[i].print();
		bc::print("------------------------------------");
	}

	bc::print("success");
	return 0;
}
