/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TENSOR_LV2_CORE_IMPL_H_
#define TENSOR_LV2_CORE_IMPL_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "Tensor_Common.h"

namespace BC {
template<class> class Tensor_Base;
namespace module {

/*
 * Defines standard utility methods related to I/O
 */

template<class derived>
struct Tensor_Utility;

template<class internal_t>
struct Tensor_Utility<Tensor_Base<internal_t>> {

	using system_tag = typename internal_t::system_tag;
    using derived = Tensor_Base<internal_t>;
    using scalar  = typename internal_t::scalar_t;
    using allocator_t = typename internal_t::allocator_t;
    using utility_l = utility::implementation<system_tag>;

    template<class>
    friend class Tensor_Utility;

private:
    static constexpr int DIMS() { return internal_t::DIMS(); }

    derived& as_derived() {
        return static_cast<derived&>(*this);
    }
    const derived& as_derived() const {
        return static_cast<const derived&>(*this);
    }
public:
    void print(int precision=8) const {
    	this->print_impl<void>(precision);
    }
private:

    static std::string format_value(const scalar& s, int precision, bool sparse=false) {
    	std::string fstr  = !sparse || std::abs(s) > .1 ? std::to_string(s) : "";
    	if (fstr.length() < (unsigned)precision)
    		return fstr.append(precision - fstr.length(), ' ');
    	else
    		return fstr.substr(0, precision);
    }

    template<class ADL=void>
    std::enable_if_t<std::is_void<ADL>::value && (DIMS() == 0)>
    print_impl(int prec) const {
    	std::cout << "[" << format_value(utility_l::extract(as_derived().memptr(), 0), prec) << "]" << std::endl;
    }

    template<class ADL=void, class v1=void>
    std::enable_if_t<std::is_void<ADL>::value && (DIMS() == 1)>
    print_impl(int prec, bool sparse=false) const {
    	std::cout << "[ ";
    	for (const auto& scalar : this->as_derived().iter()) {
    		std::cout << format_value(utility_l::extract(&scalar, 0), prec, sparse) << ", ";
    	}
    	std::cout << "]" << std::endl;
    }
    template<class ADL=void, class v1=void, class v2=void>
    std::enable_if_t<std::is_void<ADL>::value && (DIMS() > 1)>
    print_impl(int prec, bool sparse=false) const {
    	std::string dim_header;
    	dim_header.append(DIMS(), '-');

    	std::cout <<  dim_header << std::endl;
    	for (const auto slice : this->as_derived().nd_iter()) {
        	slice.print_impl(prec, sparse);
    	}
    }
public:
    void printSparse(int precision=8) const {
    	print_impl<void>(precision, true);
    }

    void write(std::ofstream& os) const {
        BC_ARRAY_ONLY("write(std::ofstream& os)");

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
        BC_ARRAY_ONLY("void write_tensor_data(std::ofstream& os)");

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
        BC_ARRAY_ONLY("void read_as_one_hot(std::ifstream& is)");

        if (derived::DIMS() != 1)
            throw std::invalid_argument("one_hot only supported by vectors");

//        as_derived().zero(); //clear FIXME COMPILE ISSUE WITH NVCC

        std::string tmp;
        std::getline(is, tmp, ',');

        as_derived()(std::stoi(tmp)) = 1;

    }
    void read(std::ifstream& is) {
        BC_ARRAY_ONLY("void read(std::ifstream& is)");

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
            utility_l::HostToDevice(as_derived().internal().memptr(), &file_data[0], (unsigned)as_derived().size() > file_data.size() ? file_data.size() : as_derived().size());

    }

    void read_tensor_data(std::ifstream& is, bool read_dimensions = true, bool overrideDimensions = true) {
        BC_ARRAY_ONLY("void read_tensor_data(std::ifstream& is, bool read_dimensions = true, bool overrideDimensions = true)");

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
            if (file_data[0] != derived::DIMS()) {
                std::cout << " attempting to read data from file of tensor of dimensions = " << file_data[0]
                        << " however the reading to tensor is of dimension = " << derived::DIMS();
                throw std::invalid_argument("Invalid Tensor File");
            }
            for (int i = 0; i < dims.size(); ++i) {
                dims[i] = file_data[i + 1];
            }
            if (overrideDimensions) {

                et::Shape<derived::DIMS()> shape;
                for (int i = 0; i < derived::DIMS(); ++i) {
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
    void read_tensor_data_as_one_hot(std::ifstream& is, int sz) {
        BC_ARRAY_ONLY("void read_tensor_data_as_one_hot(std::ifstream& is, int sz)");
        if (derived::DIMS() != 1)
            throw std::invalid_argument("one_hot only supported by vectors");

        //rescale
        if (sz > 0) {
            as_derived() = derived(sz);
        }
//                as_derived().zero(); //clear FIXME COMPILE ISSUE WITH NVCC

        std::string tmp;
        std::getline(is, tmp, ',');

        as_derived()(std::stoi(tmp)) = 1;

    }

    void print_dimensions() const {
    	if (DIMS() == 0) {
    		std::cout << "[1]" << std::endl;
    	} else {
			for (int i = 0; i < DIMS(); ++i) {
				std::cout << "[" << as_derived().dimension(i) << "]";
			}
			std::cout << std::endl;
    	}
    }
    void print_leading_dimensions() const {
        for (int i = 0; i < DIMS(); ++i) {
            std::cout << "[" << as_derived().leading_dimension(i) << "]";
        }
        std::cout << std::endl;
    }
    void print_block_dimensions() const {
        for (int i = 0; i < DIMS(); ++i) {
            std::cout << "[" << as_derived().block_dimensions(i) << "]";
        }
        std::cout << std::endl;
    }
};
}
}

#endif /* TENSOR_LV2_CORE_IMPL_H_ */
