#include "../../blackcat/tensors.h"
#include "../../blackcat/neural_networks.h"
#include "../datasets/mnist_loader.h"
#include <chrono>
#include <string>

template<class System=bc::host_tag>
int percept_MNIST(System system_tag, std::string mnist_dataset,
		int epochs=10, int batch_size=32, int samples=32*1024) {

	using value_type     = typename System::default_floating_point_type;
	using allocator_type = bc::Allocator<System, value_type>;
	using cube           = bc::Cube<value_type, allocator_type>;
	using clock          = std::chrono::duration<double>;

	auto network = bc::nn::neuralnetwork(
		bc::nn::convolution(
				system_tag,
				bc::dim(28, 28,1),  //input_image shape
				bc::dim(7, 7, 16), //krnl_rows, krnl_cols, numb_krnls
				bc::nn::adam),
		bc::nn::relu(
				system_tag,
				bc::dim(22, 22,16)),
		bc::nn::convolution(
				system_tag,
				bc::dim(22,22,16), //input_image shape
				bc::dim(7, 7, 8),  //krnl_rows, krnl_cols, numb_krnls
				bc::nn::adam),
		bc::nn::flatten(system_tag, bc::dim(16,16,8)),
		bc::nn::relu(system_tag, bc::dim(16,16,8).size()),
		bc::nn::feedforward(system_tag, bc::dim(16,16,8).size(), 64),
		bc::nn::logistic(system_tag, 64),
		bc::nn::feedforward(system_tag, 64, 10),
		bc::nn::softmax(system_tag, 10),
		bc::nn::logging_output_layer(system_tag, 10, bc::nn::RMSE).skip_every(100)
	);

	network.set_batch_size(batch_size);
	network.set_learning_rate(.03);

	bc::print("Neural Network architecture: \n",
			network.get_string_architecture());

	std::pair<cube, cube> data = bc::load_mnist(
			system_tag, mnist_dataset, batch_size, samples);

	cube& inputs = data.first;
	cube& outputs = data.second;

	bc::print("training...");
	auto start = std::chrono::system_clock::now();

	auto image_shape = bc::Dim<4>{28, 28, 1, batch_size};

	for (int i = 0; i < epochs; ++i){
		bc::print("current epoch:", i);

		for (int j = 0; j < samples/batch_size; ++j) {
			network.forward_propagation(inputs[j].reshaped(image_shape));
			network.back_propagation(outputs[j]);
			network.update_weights();
		}
	}

	auto end = std::chrono::system_clock::now();
	bc::print("training time:", clock(end - start).count());
	bc::print("testing...");

	int test_images = 10;
	auto images = inputs[0].reshaped(28,28, batch_size);

	for (int i = 0; i < test_images; ++i) {
		bc::logical(images[i].t()).print_sparse(0);
		network.single_predict(inputs[0][i].reshaped(28,28,1)).print();
		bc::print("------------------------------------");
	}


	//Test against training set (TODO split test/validation)
	int correct = 0;
	for (int i = 0; i < samples/batch_size; ++i) {
		auto hyps = network.predict(inputs[i].reshaped(image_shape));

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
