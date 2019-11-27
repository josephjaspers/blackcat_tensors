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

	static constexpr int tensor_dimension =
			ExpressionTemplate::tensor_dimension;

	derived& as_derived() {
		return static_cast<derived&>(*this);
	}

	const derived& as_derived() const {
		return static_cast<const derived&>(*this);
	}

public:

	std::string to_string(
			int precision=8,
			bool pretty=true,
			bool sparse=false) const
	{
		// TODO-to_string should not copy when the memory is allocated
		// by cudaMallocManaged. However NVCC_9.2 fails to compile
		// the code below. Ergo, cudaManaged tensors will incur a copy
		// even though this should not be the case.
		//
		//	using self_alloc_t = typename
		//		BC::traits::common_traits<ExpressionTemplate>::allocator_type;
		//	using is_managed = typename
		//	BC::allocators::allocator_traits<self_alloc_t>::is_managed_memory;

		using is_host = std::is_same<BC::host_tag, system_tag>;

#ifdef __CUDACC__
		using allocator_type = std::conditional_t<
				is_host::value,
				BC::Allocator<system_tag, value_type>,
				BC::Cuda_Managed<value_type>>;
#else
		using allocator_type = BC::Allocator<system_tag, value_type>;
#endif
		using tensor_type = Tensor_Base<exprs::Array<
				BC::Shape<tensor_dimension>,
				value_type,
				allocator_type>>;

		auto fs = BC::tensors::io::features(precision, pretty, sparse);

		static constexpr bool no_copy_required =
				/*(is_managed::value || */ is_host::value /*)*/ &&
				exprs::expression_traits<ExpressionTemplate>::is_array::value;

		return BC::traits::constexpr_ternary<no_copy_required>(
				BC::traits::bind([&](const auto& der)
				{
					return BC::tensors::io::to_string(
							der, fs, BC::traits::Integer<tensor_dimension>());
				}, as_derived()),

				BC::traits::bind([&](const auto& der)
				{
					tensor_type copy(as_derived());
					BC::device_sync();
					return BC::tensors::io::to_string(
							copy, fs, BC::traits::Integer<tensor_dimension>());
				}, as_derived()));
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

	friend std::ostream& operator << (std::ostream& os, const Tensor_Utility& self) {
		return os << self.to_string();
	}
};
}
}

#undef BC_ASSERT_ARRAY_ONLY
#endif /* TENSOR_LV2_CORE_IMPL_H_ */
