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

#include "io/Print.h"

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
    using value_type  = typename ExpressionTemplate::value_type;
    using utility_l = BC::utility::implementation<system_tag>;

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

    std::string to_string(int precision=5, bool sparse=false) const {

    	//host tensor, simple to string
		if (std::is_same<host_tag, system_tag>::value &&
				BC::tensors::exprs::expression_traits<ExpressionTemplate>::is_array) {
			return BC::tensors::io::to_string(as_derived(), precision, sparse, BC::traits::Integer<tensor_dimension>());
		}

		//if is a cuda_allocated tensor, copy it to a host_tensor
		else if (BC::tensors::exprs::expression_traits<ExpressionTemplate>::is_array) {
			using host_tensor = Tensor_Base<exprs::Array<
						tensor_dimension,
						typename ExpressionTemplate::value_type,
						BC::Allocator<host_tag, value_type>>>;

			host_tensor host_(as_derived().inner_shape());
			host_.copy(as_derived());
			return BC::tensors::io::to_string(host_, precision, sparse, BC::traits::Integer<tensor_dimension>());

			//if is an expression, evaluate to tensor, than call to_string
		} else {
			using tensor = Tensor_Base<exprs::Array<
						tensor_dimension,
						typename ExpressionTemplate::value_type,
						BC::Allocator<system_tag, value_type>>>;

			return tensor(this->as_derived()).to_string();
		}
    }

    void print(int precision=8, bool sparse=false) const {
    	std::cout << this->to_string(precision, sparse);
    }

    void print_sparse(int precision=8) const {
    	const_cast<derived&>(this->as_derived()).get_stream().sync();
    	std::cout << this->to_string(precision, true);
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

        int copy_size = (unsigned)as_derived().size() > file_data.size() ? file_data.size() : as_derived().size();
        utility_l::HostToDevice(as_derived().internal().memptr(), file_data.data(), copy_size);

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
