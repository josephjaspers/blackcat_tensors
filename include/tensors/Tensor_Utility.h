/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_UTILITY_H_
#define BLACKCAT_TENSOR_UTILITY_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

namespace BC {
namespace tensors {

template<class>
class Tensor_Base;

/*
 * Defines standard utility methods related to I/O
 */

template<class ExpressionTemplate>
struct Tensor_Utility {

	#define BC_ASSERT_ARRAY_ONLY(literal)\
	static_assert(exprs::expression_traits<ExpressionTemplate>::is_array\
			, "BC Method: '" literal "' IS NOT SUPPORTED FOR EXPRESSIONS")

	using system_tag = typename ExpressionTemplate::system_tag;
    using derived = Tensor_Base<ExpressionTemplate>;
    using scalar  = typename ExpressionTemplate::value_type;
    using utility_l = utility::implementation<system_tag>;

    template<class>
    friend class Tensor_Utility;

private:
    static constexpr int tensor_dimension = ExpressionTemplate::tensor_dimension;

    derived& as_derived() {
        return static_cast<derived&>(*this);
    }
    const derived& as_derived() const {
        return static_cast<const derived&>(*this);
    }
public:
    std::string to_string(int precision=5, bool sparse=false, bool pretty=false) const{
    	const_cast<derived&>(this->as_derived()).get_stream().sync();
    	std::stringstream ss;
    	std::string data =  this->print_impl<void>(ss, precision, sparse, pretty).str();
    	return data;
    }

    std::string to_pretty_string(int precision=5, bool sparse=false) const {
    	return to_string(precision, sparse, true);
    }

    void print(int precision=8, bool sparse=false, bool pretty=false) const {
    	std::cout << this->to_string(precision, sparse, pretty) << '\n';
    }

    void printSparse(int precision=8, bool pretty=true) const {
    	const_cast<derived&>(this->as_derived()).get_stream().sync();
    	std::cout << this->to_string(precision, true, pretty) << '\n';
    }

private:

    static std::string format_value(const scalar& s, BC::size_t precision, bool sparse=false) {
    	std::string fstr  = !sparse || std::abs(s) > .1 ? std::to_string(s) : "";
    	if (fstr.length() < (unsigned)precision)
    		return fstr.append(precision - fstr.length(), ' ');
    	else
    		return fstr.substr(0, precision);
    }
    static std::string only_if(std::string msg, bool pretty) {
    	return pretty ? msg : "";
    }

    template<class ADL=void>
    std::enable_if_t<std::is_void<ADL>::value && (tensor_dimension == 0), std::stringstream&>
    print_impl(std::stringstream& ss, int prec, bool sparse, bool pretty) const {
    	return ss << only_if("]", pretty)
    				<< format_value(utility_l::extract(as_derived().memptr(), 0), prec)
    				<< only_if("]", pretty) << '\n';
    }

    template<class ADL=void, class v1=void>
    std::enable_if_t<std::is_void<ADL>::value && (tensor_dimension == 1), std::stringstream&>
    print_impl(std::stringstream& ss, int prec, bool sparse, bool pretty) const {
    	bool first = true;

    	ss << only_if("]", pretty);
    	for (const auto& scalar : this->as_derived().iter()) {
    		if (first) {
    			first = false;
    			ss << format_value(utility_l::extract(&scalar, 0), prec, sparse);
    		} else {
    			ss << ", " << format_value(utility_l::extract(&scalar, 0), prec, sparse);
    		}
    	}
    	ss << only_if(" ]", pretty) << '\n';
    	return ss;
    }

    template<class ADL=void, class v1=void, class v2=void>
    std::enable_if_t<std::is_void<ADL>::value && (tensor_dimension == 2), std::stringstream&>
    print_impl(std::stringstream& ss, int prec, bool sparse, bool pretty) const {
    	std::string dim_header;
    	dim_header.append(tensor_dimension, '-');

    	ss <<  only_if(dim_header, pretty) << '\n';
    	auto trans = this->as_derived().transpose();
    	for (int i = 0; i < trans.dimension(1); ++i) {
        	ss << only_if("[ ", pretty);
    		bool first=true;
        	for (int j = 0; j< trans.dimension(0); ++j) {
        		if (first) {
        			first = false;
        			ss << format_value(utility_l::extract(&trans.internal()(j,i), 0), prec, sparse);
        		} else {
        			ss << ", " << format_value(utility_l::extract(&trans.internal()(j,i), 0), prec, sparse);
        		}
        		        	}
        	ss << only_if(" ]", pretty) << '\n';
    	}
    	return ss;
    }

    template<class ADL=void, class v1=void, class v2=void, class v3=void>
    std::enable_if_t<std::is_void<ADL>::value && (tensor_dimension > 2), std::stringstream&>
    print_impl(std::stringstream& ss, int prec, bool sparse, bool pretty) const {
    	std::string dim_header;
    	dim_header.append(tensor_dimension, '-');

    	ss <<  only_if(dim_header, pretty) << '\n';
    	for (const auto slice : this->as_derived().nd_iter()) {
        	slice.print_impl(ss, prec, sparse, false);
    	}
    	return ss;
    }

public:


    void write(std::ofstream& os) const {
        BC_ASSERT_ARRAY_ONLY("write(std::ofstream& os)");

        scalar* internal = new scalar[as_derived().size()];
        utility_l::DeviceToHost(internal, as_derived().internal().memptr(), as_derived().size());

        for (int i = 0; i < as_derived().size() - 1; ++i) {
            os << internal[i] << ',';
        }
        os << internal[as_derived().size() - 1]; //back
        os << '\n';

        delete[] internal;
    }

    void write_tensor_data(std::ofstream& os) const {
        BC_ASSERT_ARRAY_ONLY("void write_tensor_data(std::ofstream& os)");

        scalar* internal = new scalar[as_derived().size()];
        utility_l::DeviceToHost(internal, as_derived().internal().memptr(), as_derived().size());

        os << as_derived().dims() << ',';
        for (int i = 0; i < as_derived().dims(); ++i) {
            os << as_derived().dimension(i) << ',';
        }
        for (int i = 0; i < as_derived().size() - 1; ++i) {
            os << internal[i] << ',';
        }
        os << internal[as_derived().size() - 1]; //back
        os << '\n';

        delete[] internal;
    }

    void read_as_one_hot(std::ifstream& is) {
        BC_ASSERT_ARRAY_ONLY("void read_as_one_hot(std::ifstream& is)");

        if (derived::tensor_dimension != 1)
            throw std::invalid_argument("one_hot only supported by vectors");

        as_derived().zero();

        std::string tmp;
        std::getline(is, tmp, ',');

        as_derived()(std::stoi(tmp)) = 1;

    }

    void read_csv_row(std::ifstream& is) {
        BC_ASSERT_ARRAY_ONLY("void read(std::ifstream& is)");

        if (!is.good()) {
            std::cout << "File open error - returning " << std::endl;
            return;
        }
        std::vector<scalar> file_data;
        scalar val;
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

        int copy_size = (unsigned)as_derived().size() > file_data.size() ? file_data.size() : as_derived().size();
        utility_l::HostToDevice(as_derived().internal().memptr(), file_data.data(), copy_size);

    }

    void read_tensor_data(std::ifstream& is, bool read_dimensions = true, bool overrideDimensions = true) {
        BC_ASSERT_ARRAY_ONLY("void read_tensor_data(std::ifstream& is, bool read_dimensions = true, bool overrideDimensions = true)");

        if (!is.good()) {
            std::cout << "File open error - returning " << std::endl;
            return;
        }
        std::vector<scalar> file_data;
        scalar val;
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

        if (read_dimensions) {
            std::vector<int> dims((int) file_data[0]);
            if (file_data[0] != derived::tensor_dimension) {
                std::cout << " attempting to read data from file of tensor of dimensions = " << file_data[0]
                        << " however the reading to tensor is of dimension = " << derived::tensor_dimension;
                throw std::invalid_argument("Invalid Tensor File");
            }
            for (int i = 0; i < dims.size(); ++i) {
                dims[i] = file_data[i + 1];
            }
            if (overrideDimensions) {

                exprs::Shape<derived::tensor_dimension> shape;
                for (int i = 0; i < derived::tensor_dimension; ++i) {
                    shape.is()[i] = (int) file_data[i + 1];
                }

                as_derived() = derived(shape);
            }
            utility_l::HostToDevice(as_derived().internal().memptr(), &file_data[file_data[0] + 1],
                    as_derived().size() > file_data.size() ? file_data.size() : as_derived().size());
        } else {
            utility_l::HostToDevice(as_derived().internal().memptr(), &file_data[0],
                    as_derived().size() > file_data.size() ? file_data.size() : as_derived().size());
        }
    }

    void read_tensor_data_as_one_hot(std::ifstream& is,  BC::size_t   sz) {
        BC_ASSERT_ARRAY_ONLY("void read_tensor_data_as_one_hot(std::ifstream& is,  BC::size_t   sz)");
        if (derived::tensor_dimension != 1)
            throw std::invalid_argument("one_hot only supported by vectors");

        //rescale
        if (sz > 0) {
            as_derived() = derived(sz);
        }
		as_derived().zero();

        std::string tmp;
        std::getline(is, tmp, ',');

        as_derived()(std::stoi(tmp)) = 1;

    }

    void print_dimensions() const {
    	if (tensor_dimension == 0) {
    		std::cout << "[1]" << std::endl;
    	} else {
			for (int i = 0; i < tensor_dimension; ++i) {
				std::cout << "[" << as_derived().dimension(i) << "]";
			}
			std::cout << std::endl;
    	}
    }

    void print_leading_dimensions() const {
        for (int i = 0; i < tensor_dimension; ++i) {
            std::cout << "[" << as_derived().leading_dimension(i) << "]";
        }
        std::cout << std::endl;
    }

    void print_block_dimensions() const {
        for (int i = 0; i < tensor_dimension; ++i) {
            std::cout << "[" << as_derived().block_dimension(i) << "]";
        }
        std::cout << std::endl;
    }
};
}
}

#undef BC_ASSERT_ARRAY_ONLY
#endif /* TENSOR_LV2_CORE_IMPL_H_ */
