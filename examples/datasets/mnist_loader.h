/*
 * mnist_loader.h
 *
 *  Created on: Nov 23, 2019
 *      Author: joseph
 */

#ifndef MNIST_LOADER_H_
#define MNIST_LOADER_H_

#include "../../include/BlackCat_Tensors.h"

namespace BC {

template<class System>
auto load_mnist(System system, std::string mnist_dataset, int batch_size, int samples) {

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

	int img_sz = 784; //28 * 28 grey-scale handwritten digits
	int training_sets = samples / batch_size;

	cube inputs(img_sz, batch_size, training_sets);
	cube outputs(10, batch_size, training_sets); //10 -- 1 hot vector of outputs (10 for 1 hot digits)

	//tensor's memory is uninitialized
	inputs.zero();
	outputs.zero();

	for (int i = 0; i < training_sets && read_data.good(); ++i) {
		for (int j = 0; j < batch_size && read_data.good(); ++j) {
			outputs[i][j].read_as_one_hot(read_data); //read a single int-value (comma delimited) set that index to 1
			inputs[i][j].read_csv_row(read_data);	  //read an entire row of a csv
		 }
	}

	inputs /= 255; //Normalize (images are grey-scale values ranging 0-255)
	return std::make_pair(inputs, outputs);
}


}

#endif /* MNIST_LOADER_H_ */
