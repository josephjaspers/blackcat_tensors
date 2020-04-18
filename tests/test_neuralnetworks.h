#ifndef BC_TESTS_NEURALNETWORKS_H_
#define BC_TESTS_NEURALNETWORKS_H_

#include "test_common.h"
#include "../blackcat/neural_networks.h"


namespace bc {
namespace tests {

template<class value_type, template<class> class allocator>
int test_neuralnetworks(int sz = 128) {

	BC_TEST_BODY_HEAD

	using allocator_type = allocator<value_type>;
	using system_tag = typename bc::allocator_traits<allocator_type>::system_tag;

	BC_TEST_ON_STARTUP{};

	system_tag system;

	//Test compilation of move/copy operators
	BC_TEST_DEF(

		if (true)
			return true; 

		auto network = bc::nn::neuralnetwork(
			bc::nn::feedforward(system, 784, 256, bc::nn::momentum),
			bc::nn::tanh(system, 256),
			bc::nn::feedforward(system, 256, 10, bc::nn::momentum),
			bc::nn::softmax(system, 10),
			bc::nn::output_layer(system, 10)
		);

		network = network; 
		network = std::move(network); 
	)

	BC_TEST_DEF(

		if (true)
			return true;

		auto network = bc::nn::neuralnetwork(
			bc::nn::convolution(
				system,
				bc::dim(28, 28, 1),  //input_image shape
				bc::dim(7, 7, 16), //krnl_rows, krnl_cols, numb_krnls
				bc::nn::adam),
			bc::nn::relu(
				system,
				bc::dim(22, 22, 16)),
			bc::nn::convolution(
				system,
				bc::dim(22, 22, 16), //input_image shape
				bc::dim(7, 7, 8),  //krnl_rows, krnl_cols, numb_krnls
				bc::nn::adam),
			bc::nn::flatten(system, bc::dim(16, 16, 8)),
			bc::nn::relu(system, bc::dim(16, 16, 8).size()),
			bc::nn::feedforward(system, bc::dim(16, 16, 8).size(), 64),
			bc::nn::logistic(system, 64),
			bc::nn::feedforward(system, 64, 10),
			bc::nn::softmax(system, 10),
			bc::nn::logging_output_layer(system, 10, bc::nn::RMSE).skip_every(100)
		);

		network = network;
		network = std::move(network);
	)

	BC_TEST_DEF(
		if (true)
			return true;

		auto network = bc::nn::neuralnetwork(
			bc::nn::lstm(system, 784 / 4, 128),
			bc::nn::lstm(system, 128, 64),
			bc::nn::feedforward(system, 64, 10),
			bc::nn::softmax(system, 10),
			bc::nn::logging_output_layer(system, 10, bc::nn::RMSE).skip_every(100)
		);

		network = network;
		network = std::move(network);
	)

	BC_TEST_BODY_TAIL
}

}
}


#endif