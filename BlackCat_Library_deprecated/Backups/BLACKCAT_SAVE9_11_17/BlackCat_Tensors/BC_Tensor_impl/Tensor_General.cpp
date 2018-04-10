#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
inline void Tensor<number_type>::fill(const Scalar<number_type>& val) {
    Tensor_Operations<number_type>::fill(tensor, val(), size());
    alertUpdate();
}

template<typename number_type>
inline void Tensor<number_type>::randomize(number_type lower_bound, number_type upper_bound) {
    Tensor_Operations<number_type>::randomize(tensor, lower_bound, upper_bound, size());
    alertUpdate();
}

template<typename number_type>
inline Tensor<number_type> Tensor<number_type>::transpose() const {
    return Tensor<number_type>(*this, true, true);
}

template<typename number_type>
Tensor<number_type> Tensor<number_type>::T() const {
	return Tensor<number_type> (*this, true, true);
//	if (needsUpdate) {
//		if (self_transposed) {
//			for (int i = 0; i < totalMatrices(); ++i) {
//							unsigned index = i * matrix_size();
//
//							Tensor_Operations<number_type>::transpose(&self_transposed->tensor[index], self_transposed->leading_dimensions[1],
//							&tensor[index], rows(), cols(), leading_dimensions[1]);
//			}
//		} else {
//			self_transposed = new Tensor<number_type>(*this, true, true);
//		}
//		needsUpdate = false;
//	}
//	if (!self_transposed) {
//		self_transposed = new Tensor<number_type>(*this, true, true);
//	}
//	return *self_transposed;
}



template<typename number_type> void  Tensor<number_type>::print(const number_type* ary, const unsigned* dims, const unsigned* lead_dims, unsigned index) const {


	if (index < 3) {
		for (unsigned r = 0; r < dims[0]; ++r) {

			if (r != 0)
			std::cout << std::endl;

			for (unsigned c = 0; c< dims[1]; ++c) {
				auto str =std::to_string(ary[r + c * leading_dim(index - 1)]);
				str = str.substr(0, str.length() < 5 ? str.length() : 5);
				std::cout << str << " ";
			}
		}
		std::cout << "]" << std::endl << std::endl;

	} else {
		std::cout << "[";
		for (unsigned i = 0; i < dims[index - 1]; ++i) {
			print(&ary[i * leading_dim(index - 1)], dims, lead_dims, index - 1);
		}
	}
}


template<typename number_type> void Tensor<number_type>::print() const {

	if (!isVector()) {
	std::cout << "--- --- --- --- --- --- --- " << std::endl;
	print(tensor, ranks, ld, order);
	std::cout << "--- --- --- --- --- --- --- " << std::endl;
	} else {
		std::cout << "[";
		for (unsigned i = 0; i < sz; ++i) {
			std::cout << tensor[i] << " ";
		}
		std::cout << "]" << std::endl;
	}
}

template<typename number_type> void Tensor<number_type>::printDimensions() const {
	std::cout << "[" << rows() << "]";
	std::cout << "[" << cols() << "]";
    for (int i = 2; i < order; ++i) {
        std::cout << "[" << ranks[i] << "]";
    }
    std::cout << std::endl;
}
template<typename number_type> void Tensor<number_type>::read(std::ifstream& is) {
		reset();

		is >> order;
		is >> sz;

		ranks = new unsigned[order];
		ld = new unsigned[order];
		Tensor_Operations<number_type>::initialize(tensor, sz);
		for (unsigned i = 0; i < order; ++i) {
			is >> ranks[i];
			is >> ld[i];
		} for (unsigned i = 0; i < sz; ++i) {
			is >> tensor[i];
		}
	}

template<typename number_type> void Tensor<number_type>::readCSV(std::ifstream& is) {
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

	Tensor_Operations<number_type>::initialize(tensor, sz);

		for (unsigned i = 0; i < data.size(); ++i) {
		tensor[i] = data[i];
	}
}
template<typename number_type> void Tensor<number_type>::readCSV(std::ifstream& is, unsigned numb_values) {
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

	Tensor_Operations<number_type>::initialize(tensor, sz);

		for (unsigned i = 0; i < read_values; ++i) {
		tensor[i] = data[i];
	}
}
template<typename number_type> void Tensor<number_type>::write(std::ofstream& is) {
		is << order << ' ';
		is << sz << ' ';

		for (unsigned i = 0; i < order; ++i) {
			is << ranks[i] << ' ';
			is <<  ld[i] << ' ';
		} for (unsigned i = 0; i < sz; ++i) {
			is << tensor[i] << ' ';
		}
	}
