#include "../../include/BlackCat_Tensors.h"
#include "../../include/BlackCat_NeuralNetworks.h"
#include "../datasets/mnist_loader.h"
#include <chrono>
#include <string>

template<class System=BC::host_tag>
int percept_MNIST(System system_tag, std::string mnist_dataset,
		int epochs=10, int batch_size=32, int samples=32*1024) {

	using value_type     = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube           = BC::Cube<value_type, allocator_type>;
	using mat            = BC::Matrix<value_type, allocator_type>;
	using clock          = std::chrono::duration<double>;

	auto network = BC::nn::neuralnetwork(
		BC::nn::lstm(system_tag, 784/4, 128),
		BC::nn::lstm(system_tag, 128, 64),
		BC::nn::feedforward(system_tag, 64, 10),
		BC::nn::softmax(system_tag, 10),
		BC::nn::logging_output_layer(system_tag, 10, BC::nn::RMSE).skip_every(100)
	);

	BC::print("Neural Network architecture:");
	BC::print(network.get_string_architecture());

	network.set_learning_rate(0.03);
	network.set_batch_size(batch_size);

	std::pair<cube, cube> data = load_mnist(
			system_tag, mnist_dataset, batch_size, samples);

	cube& inputs = data.first;
	cube& outputs = data.second;

	BC::print("training...");
	auto start = std::chrono::system_clock::now();

	int img_partitions = 4;
	for (int i = 0; i < epochs; ++i){
		BC::print("current epoch:  ", i);

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
	BC::print("training time:", clock(end - start).count());

	BC::print("testing...");
	network.copy_training_data_to_single_predict(0);

	auto batch = inputs[0];
	auto shape = BC::shape(784/4, batch_size);

	for (int p = 0; p < img_partitions-1; ++p) {
		auto index = BC::index(0, 784*(p/(float)img_partitions));
		network.predict(batch[{index, shape}]);
	}

	auto last_index = BC::index(0,784*((img_partitions-1)/(float)img_partitions));
	mat hyps =network.predict(batch[{last_index, shape}]);

	BC::size_t test_images = 10;
	cube img = cube(inputs[0].reshaped(28,28, batch_size));
	for (int i = 0; i < test_images; ++i) {
		img[i].t().print_sparse(3);
		hyps[i].print();
		BC::print("------------------------------------");
	}

	BC::print("success");
	return 0;
}
