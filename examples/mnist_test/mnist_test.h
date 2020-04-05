#include "../../blackcat/tensors.h"
#include "../../blackcat/neural_networks.h"
#include "../datasets/mnist_loader.h"
#include <chrono>
#include <string>
#include "../../blackcat/tensors/tensor_static_functions.h"


template<class System=bc::host_tag>
int percept_MNIST(System system_tag, std::string mnist_dataset,
		int epochs=20, int batch_size=200, int samples=32*1024) {

	using value_type     = typename System::default_floating_point_type;
	using allocator_type = bc::Allocator<value_type, System>;
	using cube           = bc::Cube<value_type, allocator_type>;
	using mat            = bc::Matrix<value_type, allocator_type>;
	using clock          = std::chrono::duration<double>;


	auto network = bc::nn::neuralnetwork(
		bc::nn::feedforward(system_tag, 784, 256, bc::nn::momentum),
		bc::nn::tanh(system_tag, 256),
		bc::nn::feedforward(system_tag, 256, 10, bc::nn::momentum),
		bc::nn::softmax(system_tag, 10),
		bc::nn::output_layer(system_tag, 10)
	);

	network.set_batch_size(batch_size);
	network.set_learning_rate(.003);

	bc::print("Neural Network architecture:");
	bc::print(network.get_string_architecture());

	std::pair<cube, cube> data = bc::load_mnist(
			system_tag, mnist_dataset, batch_size, samples);

	cube& inputs = data.first;
	cube& outputs = data.second;

	auto train = [&]() {
		for (int i = 0; i < epochs; ++i){
			bc::print("current epoch:", i);

			for (int j = 0; j < samples/batch_size; ++j) {
				network.forward_propagation(inputs[j]);
				network.back_propagation(outputs[j]);
				network.update_weights();
			}
		}
	};

	bc::print("training...");
	auto start = std::chrono::system_clock::now();
	train();
	auto end = std::chrono::system_clock::now();
	bc::print("training time:", clock(end - start).count());

	bc::print("testing...");
	int test_images = 10;
	auto images = inputs[0].reshaped(28,28, batch_size);
	mat hyps = network.predict(inputs[0]);

	for (int i = 0; i < test_images; ++i) {
		mat(bc::logical(images[i].t())).print_sparse(0);
		hyps[i].print();
		bc::print("------------------------------------");
	}

	//Test against training set (TODO split test/validation)
	int correct = 0;
	for (int i = 0; i < samples/batch_size; ++i) {
		auto hyps = network.predict(inputs[i]);

		for (int c = 0; c < hyps.cols(); ++c) {
			if (max_index(hyps[c]) == max_index(outputs[i][c]))
				correct++;
		}
	}

	bc::print("Accuracy: ", correct, "/", inputs.size()/inputs.rows());
	bc::print('%', float(correct)/(float(inputs.size())/inputs.rows()));
	bc::print("success");
	return 0;
}
