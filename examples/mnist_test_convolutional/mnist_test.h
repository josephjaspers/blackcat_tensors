#include "../../include/BlackCat_Tensors.h"
#include "../../include/BlackCat_NeuralNetworks.h"
#include "../datasets/mnist_loader.h"
#include <chrono>
#include <string>

template<class System=BC::host_tag>
int percept_MNIST(System system_tag, std::string mnist_dataset,
		int epochs=5, int batch_size=32, int samples=32*1024) {

	using value_type     = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube           = BC::Cube<value_type, allocator_type>;
	using mat            = BC::Matrix<value_type, allocator_type>;
	using clock          = std::chrono::duration<double>;

	auto network = BC::nn::neuralnetwork(
		BC::nn::max_pooling(
				system_tag,
				BC::dim(28,28,1), //input dimension
				BC::dim(3,3),     //pool dimensions
				BC::dim(1,1)),    //pad_dimensions, out dims == (input_dims + pad_dims*2) / krnl_dims
		BC::nn::convolution(
				system_tag,
				BC::dim(10, 10,1),  //input_image shape
				BC::dim(3, 3, 16)), //krnl_rows, krnl_cols, numb_krnls
		BC::nn::relu(
				system_tag,
				BC::dim(8,8,16)),
		BC::nn::convolution(
				system_tag,
				BC::dim(8,8,16), //input_image shape
				BC::dim(3, 3, 8)),  //krnl_rows, krnl_cols, numb_krnls
		BC::nn::flatten(system_tag, BC::dim(6,6,8)),
		BC::nn::relu(system_tag, BC::dim(6,6,8).size()),
		BC::nn::feedforward(system_tag, BC::dim(6,6,8).size(), 64),
		BC::nn::tanh(system_tag, 64),
		BC::nn::feedforward(system_tag, 64, 10),
		BC::nn::softmax(system_tag, 10),
		BC::nn::logging_output_layer(system_tag, 10, BC::nn::RMSE).skip_every(100)
	);

	network.set_batch_size(batch_size);
	network.set_learning_rate(.0003);

	BC::print("Neural Network architecture: \n",
			network.get_string_architecture());

	std::pair<cube, cube> data = BC::load_mnist(
			system_tag, mnist_dataset, batch_size, samples);

	cube& inputs = data.first;
	cube& outputs = data.second;

	BC::print("training...");
	auto start = std::chrono::system_clock::now();

	auto image_shape = BC::Dim<4>{28, 28, 1, batch_size};

	for (int i = 0; i < epochs; ++i){
		BC::print("current epoch:", i);

		for (int j = 0; j < samples/batch_size; ++j) {
			network.forward_propagation(inputs[j].reshaped(image_shape));
			network.back_propagation(outputs[j]);
			network.update_weights();
		}
	}

	auto end = std::chrono::system_clock::now();
	BC::print("training time:", clock(end - start).count());
	BC::print("testing...");

	int test_images = 10;
	auto images = inputs[0].reshaped(28,28, batch_size);
	mat hyps = network.predict(inputs[0].reshaped(28,28,1, batch_size));

	for (int i = 0; i < test_images; ++i) {
		images[i].t().print_sparse(3);
		hyps[i].print();
		BC::print("------------------------------------");
	}

	BC::print("success");
	return 0;
}
