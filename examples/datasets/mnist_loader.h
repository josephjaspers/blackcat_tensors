/*
 * mnist_loader.h
 *
 *  Created on: Nov 23, 2019
 *      Author: joseph
 */

#ifndef MNIST_LOADER_H_
#define MNIST_LOADER_H_

#include "../../blackcat/tensors.h"

namespace bc {

template<class TensorView>
void read_as_one_hot(TensorView tensor, std::ifstream& is)
{
	if (TensorView::tensor_dimension != 1)
		throw std::invalid_argument("one_hot only supported by vectors");

	tensor.zero();

	std::string tmp;
	std::getline(is, tmp, ',');
	tensor(std::stoi(tmp)) = 1;
}


template<class TensorView>
void read_csv_row(TensorView tensor, std::ifstream& is)
{
	using value_type = typename TensorView::value_type;
	using system_tag = typename TensorView::system_tag;

	if (!is.good()) {
		std::cout << "File open error - returning " << std::endl;
		return;
	}
	std::vector<value_type> file_data;
	value_type val;
	std::string tmp;
	unsigned read_values = 0;

	std::getline(is, tmp, '\n');

	std::stringstream ss(tmp);

	if (ss.peek() == ',' || ss.peek() == ' ' || ss.peek() == '\t')
		ss.ignore();

	while (ss >> val) {
		file_data.push_back(val);
		++read_values;
		if (ss.peek() == ',')
			ss.ignore();
	}

	int copy_size = bc::traits::min(tensor.size(), file_data.size());
	bc::Utility<system_tag>::HostToDevice(
			tensor.data(), file_data.data(), copy_size);
}


template<class System>
auto load_mnist(System system, std::string mnist_dataset, int batch_size, int samples) {

	using value_type = typename System::default_floating_point_type;
	using allocator_type = bc::Allocator<System, value_type>;
	using cube = bc::Cube<value_type, allocator_type>;

	std::ifstream read_data(mnist_dataset);
	if (!read_data.is_open()){
		std::cout << " error opening file. " << mnist_dataset << std::endl;
		throw std::invalid_argument("Failure opening mnist_train.csv");
	}

	//move the file_ptr of the csv past the headers
	std::string tmp;
	std::getline(read_data, tmp, '\n');

	int img_sz = 784; //28 * 28 grey-scale handwritten digits
	int numb_digits = 10; //no magic numbers left behind
	int training_sets = samples / batch_size;

	cube inputs(img_sz, batch_size, training_sets);
	cube outputs(numb_digits, batch_size, training_sets);

	//tensor's memory is uninitialized
	inputs.zero();
	outputs.zero();

	for (int i = 0; i < training_sets && read_data.good(); ++i) {
		for (int j = 0; j < batch_size && read_data.good(); ++j) {
			read_as_one_hot(outputs[i][j], read_data); //read a single int-value (comma delimited) set that index to 1
			read_csv_row(inputs[i][j], read_data);	  //read an entire row of a csv
		 }
	}

	inputs /= 255; //Normalize (images are grey-scale values ranging 0-255)
	return std::make_pair(inputs, outputs);
}


}

#endif /* MNIST_LOADER_H_ */
