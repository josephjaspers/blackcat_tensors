#include "../../include/BlackCat_Tensors.h"
#include "../../include/BlackCat_NeuralNetworks.h"
#include "../datasets/mnist_loader.h"
#include <chrono>
#include <string>

template<class System=BC::host_tag>
int percept_MNIST(System system_tag, std::string mnist_dataset,
		int epochs=5, int batch_size=32, int samples=32*1024) {

	using value_type = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube = BC::Cube<value_type, allocator_type>;
	using mat  = BC::Matrix<value_type, allocator_type>;
	using clock = std::chrono::duration<double>;

	auto network = BC::nn::neuralnetwork(
		BC::nn::feedforward(system_tag, 784, 256),
		BC::nn::tanh(system_tag, 256),
		BC::nn::feedforward(system_tag, 256, 10),
		BC::nn::softmax(system_tag, 10),
		BC::nn::logging_output_layer(system_tag, 10, BC::nn::RMSE).skip_every(100)
	);

	network.set_batch_size(batch_size);
	network.set_learning_rate(.003);

	std::cout << "Neural Network architecture: \n" <<
			network.get_string_architecture() << std::endl;

	std::pair<cube, cube> data = BC::load_mnist(

			system_tag, mnist_dataset, batch_size, samples);
	cube& inputs = data.first;
	cube& outputs = data.second;

	std::cout << " training..." << std::endl;
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < epochs; ++i){
		std::cout << " current epoch: " << i << std::endl;
		for (int j = 0; j < samples/batch_size; ++j) {
			network.forward_propagation(inputs[j]);
			network.back_propagation(outputs[j]);
			network.update_weights();
		}
	}

	//Uncomment to add saving/loading
	//	network.save("mnist_test_feedforward");
	//	network.load("mnist_test_feedforward");

	auto end = std::chrono::system_clock::now();
	{
		std::cout << " testing... " << std::endl;
		int test_images = 10;
		auto images = reshape(inputs[0], BC::shape(28,28, batch_size));
		mat hyps = network.predict(inputs[0]);
		for (int i = 0; i < test_images; ++i) {
			images[i].t().print_sparse(3);
			hyps[i].print();
			std::cout << "------------------------------------" <<std::endl;
		}
	}

	{
		std::cout << " testing... " << std::endl;
		int test_images = 10;
		auto images = reshape(inputs[0], BC::shape(28,28, batch_size));
		for (int i = 0; i < test_images; ++i) {
			images[i].t().print_sparse(3);
			network.single_predict(inputs[0][i]).print();

			std::cout << "------------------------------------" <<std::endl;
		}
	}

	std::cout << " success " << std::endl;
	return 0;
}
