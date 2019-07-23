#include "../../include/BlackCat_Tensors.h" //Add BlackCat_Tensors/include/ to path
#include "../../include/BlackCat_NeuralNetworks.h" //Add BlackCat_Tensors/include/ to path

#include <chrono>

namespace {
static constexpr int default_batch_size = 32;
}

template<class System>
auto load_mnist(System system, int batch_size=default_batch_size, int samples=32*1024) {

	using value_type = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube = BC::Cube<value_type, allocator_type>;

	std::ifstream read_data("C:///Users///Rungle///blackcat_tensors///BlackCat_Tensors///examples///mnist_train.csv");
	if (!read_data.is_open()){
		std::cout << " error opening file. '..///mnist_train.csv'"
				"Program expected to be run from examples directory." << std::endl;
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
int percept_MNIST(System system_tag=System(), int epochs=10, int batch_size=default_batch_size, int samples=32*1024) {

	using namespace BC::nn;

	using value_type = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube = BC::Cube<value_type, allocator_type>;
	using mat  = BC::Matrix<value_type, allocator_type>;
	using clock = std::chrono::duration<double>;


    auto network = neuralnetwork(
    		feedforward(system_tag, 784, 256),
    			logistic(system_tag, 256),
    			feedforward(system_tag, 256, 10),
    			softmax(system_tag, 10),
    			outputlayer(system_tag, 10)
    );
    network.set_batch_size(batch_size);

    std::pair<cube, cube> data = load_mnist(system_tag, batch_size, samples);
    cube& inputs = data.first;
    cube& outputs = data.second;
	std::cout << " training..." << std::endl;


	auto start = std::chrono::system_clock::now();

	for (int i = 0; i < epochs; ++i){
		std::cout << " current epoch: " << i++ << std::endl;
		for (int j = 0; j < samples/batch_size; ++j) {

				network.forward_propagation(inputs[j]);
				network.back_propagation(outputs[j]);
				network.update_weights();
		}
	}

	auto end = std::chrono::system_clock::now();
	clock total = clock(end - start);
	std::cout << " training time: " <<  total.count() << std::endl;
	std::cout << " testing... " << std::endl;

	BC::size_t test_images = 10;
	cube img = cube(reshape(inputs[0])(28,28, batch_size));
	mat hyps = mat(network.forward_propagation(inputs[0]));

	for (int i = 0; i < test_images; ++i) {
		img[i].printSparse(3);
		hyps[i].print();
		std::cout << "------------------------------------" <<std::endl;
	}
	std::cout << " success " << std::endl;
	return 0;
}
