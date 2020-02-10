#include "../../blackcat/tensors.h"
#include "../../blackcat/neural_networks.h"
#include "../datasets/mnist_loader.h"
#include <chrono>
#include <string>
#include "../../blackcat/tensors/tensor_static_functions.h"


template<class System=bc::host_tag>
int percept_MNIST(System system_tag, std::string mnist_dataset,
		int epochs=10, int batch_size=200, int samples=32*1024) {

	using value_type     = typename System::default_floating_point_type;
	using allocator_type = bc::Allocator<System, value_type>;
	using cube           = bc::Cube<value_type, allocator_type>;
	using mat            = bc::Matrix<value_type, allocator_type>;
	using clock          = std::chrono::duration<double>;


//	auto network = bc::nn::neuralnetwork(
//		bc::nn::feedforward(system_tag, 784, 256, bc::nn::momentum),
//		bc::nn::tanh(system_tag, 256),
//		bc::nn::feedforward(system_tag, 256, 10, bc::nn::momentum),
//		bc::nn::softmax(system_tag, 10),
//		bc::nn::logging_output_layer(system_tag, 10, bc::nn::RMSE).skip_every(100)
//	);

	using ff = bc::nn::FeedForward<bc::host_tag, double, bc::nn::Stochastic_Gradient_Descent>;

	using sm = bc::nn::SoftMax<bc::host_tag, double>;
	using il = bc::nn::Input_Layer<bc::host_tag, double, bc::traits::Integer<1>>;
	using smnode = std::shared_ptr<sm>;
	using ffnode = std::shared_ptr<ff>;
	using ilnode = std::shared_ptr<il>;
	using node = std::shared_ptr<bc::nn::Layer_Base<bc::traits::Integer<1>, double, bc::host_tag>>;

	node input = node(new il(bc::dim(784)));
	node a = node(new ff(10));
//	node b = node(new ff(10));
	node softmax = node(new sm(10));

	bc::print("setting");
//	input->link(a);
//	a->link(b);
//	b->link(softmax);
	link(input, a);
//	link(a,b);
	link(a,softmax);


	input->set_batch_size(batch_size);
	a->set_batch_size(batch_size);
//	b->set_batch_size(batch_size);
	softmax->set_batch_size(batch_size);

	bc::print("init");
	input->init();
	a->init();
//	b->init();
	softmax->init();



	bc::print("set batch and lr");
	auto network = a;


	a->set_learning_rate(.003);
//	b->set_learning_rate(.003);

	bc::print("Neural Network architecture:");
//	bc::print(network->get_string_architecture());

	bc::print("loading data");
	std::pair<cube, cube> data = bc::load_mnist(
			system_tag, mnist_dataset, batch_size, samples);

	cube& inputs = data.first;
	cube& outputs = data.second;

	auto fp = [&](const auto& X) {
		return a->fp(input->fp(X));
	};

	auto bp = [&](const auto& dy) {
		return a->bp(dy);
	};

	auto train = [&]() {
		for (int i = 0; i < epochs; ++i){
			bc::print("current epoch:", i);

			for (int j = 0; j < samples/batch_size; ++j) {
				bp(fp(inputs[j]) - outputs[j]);
				a->update_weights();
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
	auto hyps = fp(inputs[0]);

	for (int i = 0; i < test_images; ++i) {
		bc::logical(images[i].t()).print_sparse(0);
		hyps[i].print();
		bc::print("------------------------------------");
	}

	//Test against training set (TODO split test/validation)
	int correct = 0;
	for (int i = 0; i < samples/batch_size; ++i) {
//		auto hyps = softmax->fp(b->fp(a->fp(inputs[i])));
		auto htps = fp(inputs[i]);
//		auto hyps = network.predict(inputs[i]);

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
