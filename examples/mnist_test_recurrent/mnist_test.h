#include "../../include/BlackCat_Tensors.h" //Add BlackCat_Tensors/include/ to path
#include "../../include/BlackCat_NeuralNetworks.h" //Add BlackCat_Tensors/include/ to path

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

	int img_sz = 784; //28 * 28 greyscale handwritten digits
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

	using namespace BC::nn;

	using value_type = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube = BC::Cube<value_type, allocator_type>;
	using mat  = BC::Matrix<value_type, allocator_type>;
	using clock = std::chrono::duration<double>;

#ifdef BC_MNIST_TEST_LSTM
	epochs = 1000; //Requires more iterations to train
	samples = 128; //reduce the number of samples (this takes much longer to train)

    auto network = neuralnetwork(
				feedforward(system_tag, 784/4, 64),
				tanh(system_tag, 64),
    			lstm(system_tag, 64, 10),
    			softmax(system_tag, 10),
    			outputlayer(system_tag, 10));
#else
    auto network = neuralnetwork(
    			feedforward(system_tag, 784/4, 128),
    			tanh(system_tag, 128),
    			recurrent(system_tag, 128, 64),
    			tanh(system_tag, 64),
    			recurrent(system_tag, 64, 10),
    			softmax(system_tag, 10),
    			outputlayer(system_tag, 10)
    );
#endif
    network.set_batch_size(batch_size);

    std::pair<cube, cube> data = load_mnist(system_tag, mnist_dataset, batch_size, samples);
    cube& inputs = data.first;
    cube& outputs = data.second;

	std::cout << " training..." << std::endl;
	auto start = std::chrono::system_clock::now();

	for (int i = 0; i < epochs; ++i){
		std::cout << " current epoch: " << i << std::endl;
		for (int j = 0; j < samples/batch_size; j++) {

				//Feed the top half of the image followed by the bottom half
				network.forward_propagation(chunk(inputs[j], BC::index(0,        0), BC::shape(784/4, batch_size)));
				network.forward_propagation(chunk(inputs[j], BC::index(784* 1/4, 0), BC::shape(784/4, batch_size)));
				network.forward_propagation(chunk(inputs[j], BC::index(784* 2/4, 0), BC::shape(784/4, batch_size)));
				network.forward_propagation(chunk(inputs[j], BC::index(784* 3/4, 0), BC::shape(784/4, batch_size)));


				//Truncate BP to only the last output
				network.back_propagation(outputs[j]);
				network.back_propagation(outputs[j]);

				network.update_weights();
		}
	}

	auto end = std::chrono::system_clock::now();
	clock total = clock(end - start);
	std::cout << " training time: " <<  total.count() << std::endl;

	std::cout << " testing... " << std::endl;
	BC::size_t test_images = 10;
	cube img = cube(reshape(inputs[0], BC::shape(28,28, batch_size)));
	network.forward_propagation(chunk(inputs[0], BC::index(0,        0), BC::shape(784/4, batch_size)));
	network.forward_propagation(chunk(inputs[0], BC::index(784* 1/4, 0), BC::shape(784/4, batch_size)));
	network.forward_propagation(chunk(inputs[0], BC::index(784* 2/4, 0), BC::shape(784/4, batch_size)));
	mat hyps =network.forward_propagation(chunk(inputs[0], BC::index(784* 3/4, 0), BC::shape(784/4, batch_size)));

	for (int i = 0; i < test_images; ++i) {
		mat(img[i].t()).print_sparse(3);
		hyps[i].print();
		std::cout << "------------------------------------" <<std::endl;
	}
	std::cout << " success " << std::endl;
	return 0;
}
