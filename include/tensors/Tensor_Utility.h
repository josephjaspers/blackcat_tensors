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
	static_assert(exprs::expression_traits<ExpressionTemplate>::is_array::value\
			, "BC Method: '" literal "' IS NOT SUPPORTED FOR EXPRESSIONS")

	using system_tag = typename ExpressionTemplate::system_tag;
	using derived = Tensor_Base<ExpressionTemplate>;
	using value_type  = typename ExpressionTemplate::value_type;

	template<class>
	friend struct Tensor_Utility;

private:

	static constexpr int tensor_dimension = ExpressionTemplate::tensor_dimension;

	derived& as_derived() {
		return static_cast<derived&>(*this);
	}

	const derived& as_derived() const {
		return static_cast<const derived&>(*this);
	}

	//If host_tensor
	template<template<int> class Integer>
	std::string to_string(Integer<0>, BC::tensors::io::features fs) const {
		return BC::tensors::io::to_string(as_derived(), fs, BC::traits::Integer<tensor_dimension>());
	}
	//If device_tensor
	template<template<int> class Integer>
	std::string to_string(Integer<1>, BC::tensors::io::features fs) const {
		using host_tensor = Tensor_Base<exprs::Array<
								BC::Shape<tensor_dimension>,
								typename ExpressionTemplate::value_type,
								BC::Allocator<host_tag, value_type>>>;

		host_tensor host_(BC::Shape<tensor_dimension>(as_derived().get_shape()));
		host_.copy(as_derived());
		return BC::tensors::io::to_string(host_, fs, BC::traits::Integer<tensor_dimension>());
	}

	template<template<int> class Integer>
	std::string to_string(Integer<-1>, BC::tensors::io::features fs) const {
		using host_tensor = Tensor_Base<exprs::Array<
								BC::Shape<tensor_dimension>,
								typename ExpressionTemplate::value_type,
								BC::Allocator<host_tag, value_type>>>;

					host_tensor host_;
					host_.copy(as_derived());
					return BC::tensors::io::to_string(host_, fs, BC::traits::Integer<tensor_dimension>());
	}


	//If expression_type
	template<template<int> class Integer>
	std::string to_string(Integer<2>, BC::tensors::io::features fs) const {
		using tensor = Tensor_Base<exprs::Array<
					BC::Shape<tensor_dimension>,
					typename ExpressionTemplate::value_type,
					BC::Allocator<system_tag, value_type>>>;

		return tensor(this->as_derived()).to_string(fs.precision, fs.pretty, fs.sparse);
	}

public:

	std::string to_string(int precision=8, bool pretty=true, bool sparse=false) const {
		using specialization =
				std::conditional_t<
					BC::tensors::exprs::expression_traits<ExpressionTemplate>::is_expr::value, BC::traits::Integer<2>,
				std::conditional_t<
					std::is_same<host_tag, system_tag>::value, BC::traits::Integer<0>,
				BC::traits::Integer<1>>>;

			return this->to_string(specialization(), BC::tensors::io::features(precision, pretty, sparse));
		}

	std::string to_raw_string(int precision=8) const {
		return this->to_string(precision, false, false);
	}

	void print(int precision=8, bool pretty=true, bool sparse=false) const {
		std::cout << this->to_string(precision, pretty, sparse) << std::endl;
	}

	void print_sparse(int precision=8, bool pretty=true) const {
		std::cout << this->to_string(precision, pretty, true) << std::endl;
	}

	void raw_print(int precision=0, bool sparse=false) const {
		std::cout << this->to_string(precision, false, sparse) << std::endl;
	}

	//TODO deprecate this
	void read_as_one_hot(std::ifstream& is) {
		BC_ASSERT_ARRAY_ONLY("void read_as_one_hot(std::ifstream& is)");

		if (derived::tensor_dimension != 1)
			throw std::invalid_argument("one_hot only supported by vectors");

		as_derived().zero();

		std::string tmp;
		std::getline(is, tmp, ',');

		as_derived()(std::stoi(tmp)) = 1;
	}

	//TODO deprecate this
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
		BC::utility::implementation<system_tag>::HostToDevice(as_derived().internal().data(), file_data.data(), copy_size);

	}

	void print_dimensions() const {
		for (int i = 0; i < tensor_dimension; ++i) {
			std::cout << "[" << as_derived().dimension(i) << "]";
		}
		std::cout << std::endl;
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
