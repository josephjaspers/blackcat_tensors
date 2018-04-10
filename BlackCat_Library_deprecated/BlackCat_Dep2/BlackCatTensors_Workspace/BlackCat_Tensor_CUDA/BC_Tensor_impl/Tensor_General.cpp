#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type, class TensorOperations>
inline void Tensor<number_type, TensorOperations>::fill(const Scalar<number_type, TensorOperations>& val) {
    CPU::fill(tensor, ranks, order, ld, val());
    alertUpdate();
}

template<typename number_type, class TensorOperations>
inline void Tensor<number_type, TensorOperations>::randomize(number_type lower_bound, number_type upper_bound) {
    CPU::randomize(tensor, lower_bound, upper_bound, size());
    alertUpdate();
}

template<typename number_type, class TensorOperations>
number_type Tensor<number_type, TensorOperations>::max() const {
	number_type max_val = tensor[0];
    CPU::max(&max_val, tensor, ranks, ld, order);
    return max_val;
}

template<typename number_type, class TensorOperations>
number_type Tensor<number_type, TensorOperations>::min() const {
	number_type min_val = tensor[0];

    CPU::min(&min_val, tensor, ranks, ld, order);
    return min_val;
}

template<typename number_type, class TensorOperations>
std::pair<number_type, Tensor<unsigned, TensorOperations>> Tensor<number_type, TensorOperations>::max_index() const {
	std::pair<number_type, Tensor<unsigned, TensorOperations>> dataHolder;

	Tensor<unsigned, TensorOperations> indexes(order);
	number_type max_val;

	CPU::max_index(&max_val, indexes.data(), tensor, ranks, ld, order);

	dataHolder.first = max_val;
	dataHolder.second = indexes;

    return dataHolder;
}

template<typename number_type, class TensorOperations>
std::pair<number_type, Tensor<unsigned, TensorOperations>> Tensor<number_type, TensorOperations>::min_index() const {
	std::pair<number_type, Tensor<unsigned, TensorOperations>> dataHolder;

	Tensor<unsigned, TensorOperations> indexes(order);
	number_type max_val;

	CPU::min_index(&max_val, indexes.data(), tensor, ranks, ld, order);

	dataHolder.first = max_val;
	dataHolder.second = indexes;

    return dataHolder;
}

template<typename number_type, class TensorOperations>
inline Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::transpose() const {
    return Tensor<number_type, TensorOperations>(*this, true, true);
}

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::T() const {
	return Tensor<number_type, TensorOperations> (*this, true, true);
//	if (needsUpdate) {
//		if (self_transposed) {
//			for (int i = 0; i < totalMatrices(); ++i) {
//							unsigned index = i * matrix_size();
//
//							Tensor_Operations<number_type>::transpose(&self_transposed->tensor[index], self_transposed->leading_dimensions[1],
//							&tensor[index], rows(), cols(), leading_dimensions[1]);
//			}
//		} else {
//			self_transposed = new Tensor<number_type, TensorOperations>(*this, true, true);
//		}
//		needsUpdate = false;
//	}
//	if (!self_transposed) {
//		self_transposed = new Tensor<number_type, TensorOperations>(*this, true, true);
//	}
//	return *self_transposed;
}

template<typename number_type, class TensorOperations> void Tensor<number_type, TensorOperations>::print() const {

	if (!isVector()) {
	std::cout << "--- --- --- --- --- --- --- " << std::endl;
	CPU::print(tensor, ranks, ld, order);
	std::cout << "--- --- --- --- --- --- --- " << std::endl;
	} else {
		std::cout << "[";
		for (unsigned i = 0; i < sz; ++i) {
			std::cout << tensor[i] << " ";
		}
		std::cout << "]" << std::endl;
	}
}

//template<typename number_type, class TensorOperations> void Tensor<number_type, TensorOperations>::printDimensions() const {
//	std::cout << "[" << rows() << "]";
//	std::cout << "[" << cols() << "]";
//    for (int i = 2; i < order; ++i) {
//        std::cout << "[" << ranks[i] << "]";
//    }
//    std::cout << std::endl;
//    std::cout << std::endl;
//}
template<typename number_type, class TensorOperations> void Tensor<number_type, TensorOperations>::read(std::ifstream& is) {
		reset();

		is >> order;
		is >> sz;

		ranks = new unsigned[order];
		ld = new unsigned[order];
		CPU::initialize(tensor, sz);
		for (unsigned i = 0; i < order; ++i) {
			is >> ranks[i];
			is >> ld[i];
		} for (unsigned i = 0; i < sz; ++i) {
			is >> tensor[i];
		}
	}

template<typename number_type, class TensorOperations> void Tensor<number_type, TensorOperations>::readCSV(std::ifstream& is) {
		if (!is.good()) {
			std::cout <<"File open error - returning " << std::endl;
			return;
		}
		reset();

		std::vector<double> data;
		while (is.good())
		{
			std::string tmp;
		    std::getline(is, tmp, '\n');

		    std::stringstream ss(tmp);
		    double val;
		    while (ss >> val)
		    {
		    data.push_back(val);
	       if (ss.peek() == ',')
	           ss.ignore();
	    }
	}

	order = 1;
	sz = data.size();
	ranks = new unsigned[1];
	ranks[0] = data.size();
	ld = new unsigned[1];
	ld[0] = data.size();

	CPU::initialize(tensor, sz);

		for (unsigned i = 0; i < data.size(); ++i) {
		tensor[i] = data[i];
	}
}
template<typename number_type, class TensorOperations> void Tensor<number_type, TensorOperations>::readCSV(std::ifstream& is, unsigned numb_values) {
		if (!is.good()) {
			std::cout <<"File open error - returning " << std::endl;
			return;
		}
		reset();

		std::vector<double> data;

		unsigned read_values = 0;
		while (is.good() && read_values != numb_values)
		{
			std::string tmp;
		    std::getline(is, tmp, '\n');

		    std::stringstream ss(tmp);
		    double val;

		    if (ss.peek() == ',')
		   ss.ignore();

		    while (ss >> val)
		    {
		    	data.push_back(val);
		    	++read_values;
		    	if (ss.peek() == ',')
		    		ss.ignore();
	    }
	}

	order = 1;
	sz = read_values;
	ranks = new unsigned[1];
	ranks[0] = read_values;
	ld = new unsigned[1];
	ld[0] = 1;

	CPU::initialize(tensor, sz);

		for (unsigned i = 0; i < read_values; ++i) {
		tensor[i] = data[i];
	}
}
template<typename number_type, class TensorOperations> void Tensor<number_type, TensorOperations>::write(std::ofstream& is) {
		is << order << ' ';
		is << sz << ' ';

		for (unsigned i = 0; i < order; ++i) {
			is << ranks[i] << ' ';
			is <<  ld[i] << ' ';
		} for (unsigned i = 0; i < sz; ++i) {
			is << tensor[i] << ' ';
		}
	}
