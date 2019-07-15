#include "BlackCat_NeuralNetworks.h"

#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>

namespace BC {
namespace nn {

template<class System=BC::host_tag>
int percept_MNIST(System system_tag=System()) {

	using value_type = typename System::default_floating_point_type;
	using allocator_type = BC::Allocator<System, value_type>;
	using cube = BC::Cube<value_type, allocator_type>;
	using mat  = BC::Matrix<value_type, allocator_type>;
	using clock = std::chrono::duration<double>;

	std::string fname = "mnist_train.csv";
    BC::size_t data_size = 1024 * 32;
    BC::size_t batch_size = 32;
    BC::size_t picture_size = 784;
    BC::size_t numb_batches = data_size / batch_size;
    BC::size_t epochs = 10;
    BC::size_t numb_digits = 10;

    auto network = neuralnetwork(
    			feedforward(system_tag, 784, 256),
    			logistic(system_tag, 256),
    			feedforward(system_tag, 256, 10),
    			logistic(system_tag, 10),
    			outputlayer(system_tag, 10)
    );

    network.set_batch_size(batch_size);

    cube inputs(picture_size, batch_size, numb_batches);
    cube outputs(numb_digits, batch_size, numb_batches);

    //Load the data ---------------------------------
	inputs.zero();
	outputs.zero();

	std::cout << "loading data..." << std::endl << std::endl;
	std::ifstream in_stream("///home/joseph///Datasets///all///train.csv");

	if (!in_stream.is_open()){
		std::cout << " error opening file" << std::endl;
		return -1;
	}

	//move the file_ptr of the csv past the headers
	std::string tmp; std::getline(in_stream, tmp, '\n');

	std::cout << " generating loading " << std::endl;
	for (int i = 0; i < numb_batches && in_stream.good(); ++i) {
		for (int j = 0; j < batch_size && in_stream.good(); ++j) {
			outputs[i][j].read_as_one_hot(in_stream);
			inputs[i][j].read_csv_row(in_stream);
		 }
	}
	inputs = BC::normalize(inputs, 0, 255);



	//Train -------------------------------------------------
    std::cout << " training..." << std::endl;
    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < epochs; ++i) {
        std::cout << " current epoch: " << i << std::endl;
        for (int j = 0; j < numb_batches; ++j) {
            network.forward_propagation(inputs[j]);
            network.back_propagation(outputs[j]);
            network.update_weights();
        }
    }

    auto end = std::chrono::system_clock::now();
    clock total = clock(end - start);
    std::cout << " training time: " <<  total.count() << std::endl;

    std::cout << "success \n\n" << std::endl;

    {

    //Test -------------------------------------------------

    std::cout << " testing... " << std::endl;
    BC::size_t test_images = 10;
    cube img = cube(reshape(inputs[0])(28,28, batch_size));
    mat hyps = mat(network.forward_propagation(inputs[0]));

    for (int i = 0; i < test_images; ++i) {
        img[i].printSparse(3);
        hyps[i].print();
        std::cout << "------------------------------------" <<std::endl;
    }

    return 0;
}
}
}
}
