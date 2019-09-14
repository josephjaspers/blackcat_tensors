/*
 * Function_Conv2d.h
 *
 *  Created on: Aug 21, 2019
 *	  Author: joseph
 */

#ifndef FUNCTION_CONV2D_H_
#define FUNCTION_CONV2D_H_

#include "functions/convolutions/Convolution.h"
#include "Expression_Template_Base.h"
#include "Tree_Evaluator.h"

namespace BC {
namespace tensors {
namespace exprs {

//Not actually a BLAS function but we want the optimizer to treat it as if it was one
struct multichannel_conv2d {
	using requires_greedy_evaluation = std::true_type;
};

struct multichannel_conv2d_data_backwards {
	using requires_greedy_evaluation = std::true_type;
};

struct multichannel_conv2d_kernel_backwards {
	using requires_greedy_evaluation = std::true_type;
};

struct img2col {
	using requires_greedy_evaluation = std::true_type;
};


template<class lv, class rv>
struct Binary_Expression<multichannel_conv2d, lv, rv>
: Expression_Base<Binary_Expression<multichannel_conv2d, lv, rv>>, multichannel_conv2d {

	static_assert((lv::tensor_dimension == 3 || lv::tensor_dimension==4),"Kernel must have 3 or 4 dimensions");
	static_assert(rv::tensor_dimension==3, "Image (rv) dimension must be 3");

	using value_type = typename lv::value_type;
	using system_tag = typename lv::system_tag;
	using blas_impl  = BC::blas::implementation<system_tag>;

	static constexpr int tensor_dimension  = 3;
	static constexpr int tensor_iterator_dimension = 3;

	lv left;
	rv right;

	BC::size_t stride = 1;
	BC::size_t padding = 0;

	BCINLINE BC::size_t  size() const { return rows() * cols() * this->dimension(2); }
	BCINLINE BC::size_t  rows() const { return right.rows() - left.rows() + padding*2 + 1;  }
	BCINLINE BC::size_t  cols() const { return right.cols() - left.cols() + padding*2 + 1; }
	BCINLINE BC::size_t  dimension(int i) const {
		if (i == 0)
			return rows();
		else if (i == 1)
			return cols();
		else if (i == 2)
			return left.dimension(3);
		else
			return 1;
	}

	Binary_Expression(lv left, rv right, BC::size_t stride_, BC::size_t padding_):
		left(left), right(right), stride(stride_), padding(padding_) {}

	Binary_Expression(lv left, rv right, multichannel_conv2d default_=multichannel_conv2d()):
		left(left), right(right), stride(1), padding(0) {}


	template<class core, int alpha_mod, int beta_mod, class Stream>
	void eval(injector<core, alpha_mod, beta_mod> injection_values, Stream stream) const {

		for (int i = 0; i < tensor_dimension; ++i) {
			BC_ASSERT(dimension(0) == injection_values.data().dimension(0),
					"INVALID TENSOR INJECTION INCORRECT DIMENSIONS");
		}

		//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
		auto& injection = injection_values.data();

		//greedy evaluate the whole expression, currently we do not support transposition/scalars/etc
		auto left_evaluated = greedy_evaluate(left, stream);
		auto right_evaluated = greedy_evaluate(right, stream);

		//call convolution (technically correlation)
		BC::tensors::exprs::functions::conv2d(
				stream,
				injection,
				left, right,
				padding, stride,
				alpha_mod, beta_mod);

		//deallocate if need be
		if (expression_traits<decltype(left_evaluated)>::is_temporary::value) {
			using vt = typename decltype(left_evaluated)::value_type;

			stream.template get_allocator_rebound<vt>().deallocate(left_evaluated.memptr(), left_evaluated.size());
		}
		if (expression_traits<decltype(right_evaluated)>::is_temporary::value) {
			using vt = typename decltype(right_evaluated)::value_type;

			stream.template get_allocator_rebound<vt>().deallocate(right_evaluated.memptr(), right_evaluated.size());
		}
	}
};



template<class lv, class rv>
struct Binary_Expression<multichannel_conv2d_kernel_backwards, lv, rv>:
	Expression_Base<Binary_Expression<multichannel_conv2d_kernel_backwards, lv, rv>>, multichannel_conv2d_kernel_backwards {

	using value_type = typename lv::value_type;
	using system_tag = typename lv::system_tag;
	using blas_impl  = BC::blas::implementation<system_tag>;

	static constexpr int tensor_dimension  = 4;
	static constexpr int tensor_iterator_dimension = 4;

	lv left;
	rv right;

	BC::size_t stride = 1;
	BC::size_t padding = 0;

	BCINLINE
	BC::size_t size() const {
		return rows() * cols() * dimension(2) * dimension(3);
	}
	BCINLINE BC::size_t  rows() const { return left.rows() - right.rows() + padding*2 + 1;  }
	BCINLINE BC::size_t  cols() const { return left.cols() - right.cols() + padding*2 + 1; }
	BCINLINE BC::size_t  dimension(int i) const {
		if (i == 0)
			return rows();
		else if (i == 1)
			return cols();
		else if (i == 2)
			return left.dimension(2);
		else if (i == 3)
			return right.dimension(2);
		else
			return 1;
	}

	Binary_Expression(lv left, rv right, BC::size_t stride_, BC::size_t padding_):
		left(left), right(right), stride(stride_), padding(padding_) {}

	Binary_Expression(lv left, rv right, multichannel_conv2d_kernel_backwards default_=multichannel_conv2d_kernel_backwards()):
		left(left), right(right), stride(1), padding(0) {}


	template<class core, int alpha_mod, int beta_mod, class Stream>
	void eval(injector<core, alpha_mod, beta_mod> injection_values, Stream stream) const {

		for (int i = 0; i < tensor_dimension; ++i) {
			BC_ASSERT(dimension(0) == injection_values.data().dimension(0),
					"INVALID TENSOR INJECTION INCORRECT DIMENSIONS");
		}

		//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
		auto& injection = injection_values.data();

		//greedy evaluate the whole expression, currently we do not support transposition/scalars/etc
		auto left_evaluated = greedy_evaluate(left, stream);
		auto right_evaluated = greedy_evaluate(right, stream);

		//call convolution (technically correlation)
		BC::tensors::exprs::functions::conv2d_kernel_backwards(
				stream,
				injection,
				left, right,
				padding, stride,
				alpha_mod,
				beta_mod);

		//deallocate if need be
		if (expression_traits<decltype(left_evaluated)>::is_temporary::value) {
			using vt = typename decltype(left_evaluated)::value_type;

			stream.template get_allocator_rebound<vt>().deallocate(left_evaluated.memptr(), left_evaluated.size());
		}
		if (expression_traits<decltype(right_evaluated)>::is_temporary::value) {
			using vt = typename decltype(right_evaluated)::value_type;

			stream.template get_allocator_rebound<vt>().deallocate(right_evaluated.memptr(), right_evaluated.size());
		}
	}
};


template<class lv, class rv>
struct Binary_Expression<multichannel_conv2d_data_backwards, lv, rv>
: Expression_Base<Binary_Expression<multichannel_conv2d_data_backwards, lv, rv>>, multichannel_conv2d_data_backwards {

	using value_type = typename lv::value_type;
	using system_tag = typename lv::system_tag;
	using blas_impl  = BC::blas::implementation<system_tag>;

	static constexpr int tensor_dimension  = 3;
	static constexpr int tensor_iterator_dimension = 3;

	lv left;
	rv right;

	BC::size_t stride = 1;
	BC::size_t padding = 0;

	BCINLINE BC::size_t  size() const { return rows() * cols() * this->dimension(2); }
	BCINLINE BC::size_t  rows() const { return right.rows() + left.rows() - padding*2 - 1;  }
	BCINLINE BC::size_t  cols() const { return right.cols() + left.cols() - padding*2 - 1; }
	BCINLINE BC::size_t  dimension(int i) const {
		if (i == 0)
			return rows();
		else if (i == 1)
			return cols();
		else if (i == 2)
			return left.dimension(2);
		else
			return 1;
	}

	Binary_Expression(lv left, rv right, BC::size_t stride_, BC::size_t padding_):
		left(left), right(right), stride(stride_), padding(padding_) {}

	Binary_Expression(lv left, rv right, multichannel_conv2d default_=multichannel_conv2d()):
		left(left), right(right), stride(1), padding(0) {}


	template<class core, int alpha_mod, int beta_mod, class Stream>
	void eval(injector<core, alpha_mod, beta_mod> injection_values, Stream stream) const {

		for (int i = 0; i < tensor_dimension; ++i) {
			BC_ASSERT(dimension(0) == injection_values.data().dimension(0),
					"INVALID TENSOR INJECTION INCORRECT DIMENSIONS");
		}

		//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
		auto& injection = injection_values.data();

		//greedy evaluate the whole expression, currently we do not support transposition/scalars/etc
		auto left_evaluated = greedy_evaluate(left, stream);
		auto right_evaluated = greedy_evaluate(right, stream);

		//call convolution (technically correlation)
		BC::tensors::exprs::functions::conv2d_data_backwards(
				stream,
				injection,
				left, right,
				padding, stride,
				alpha_mod, beta_mod);

		//deallocate if need be
		if (expression_traits<decltype(left_evaluated)>::is_temporary::value) {
			using vt = typename decltype(left_evaluated)::value_type;

			stream.template get_allocator_rebound<vt>().deallocate(left_evaluated.memptr(), left_evaluated.size());
		}
		if (expression_traits<decltype(right_evaluated)>::is_temporary::value) {
			using vt = typename decltype(right_evaluated)::value_type;

			stream.template get_allocator_rebound<vt>().deallocate(right_evaluated.memptr(), right_evaluated.size());
		}
	}
};

template<class Array>
struct Unary_Expression<img2col, Array>
: Expression_Base<Unary_Expression<img2col, Array>>, img2col {

	using value_type = typename Array::value_type;
	using system_tag = typename Array::system_tag;
	using blas_impl  = BC::blas::implementation<system_tag>;

	static_assert(Array::tensor_dimension>=3, "img2col expects at least a cube");
	static constexpr int tensor_dimension  = Array::tensor_dimension-1;
	static constexpr int tensor_iterator_dimension = tensor_dimension;

	Array array;

	BC::size_t stride = 1;
	BC::size_t padding = 0;
	BC::Shape<3> krnl_shape;

	BCINLINE BC::size_t size() const {
		return rows() * cols() * dimension(2) * dimension(3);
	}

	BCINLINE BC::size_t  rows() const { return dimension(0); }
	BCINLINE BC::size_t  cols() const { return dimension(1); }
	BCINLINE BC::size_t  dimension(int i) const {
		if (i == 0)
			return krnl_shape.size();
		else if (i == 1) //number of krnl_positions
			return ((array.dimension(0) - krnl_shape.dimension(0) + padding*2)/stride) *
					 ((array.dimension(1) - krnl_shape.dimension(1) + padding*2)/stride);
		else if (i == 2)
			return array.dimension(3); //batch_size
		else
			return 1;
	}

	Unary_Expression(Array array, BC::Shape<3> krnl_shape, BC::size_t stride_, BC::size_t padding_):
		array(array), krnl_shape(krnl_shape), stride(stride_), padding(padding_) {}

	template<class core, int alpha_mod, int beta_mod, class Stream>
	void eval(injector<core, alpha_mod, beta_mod> injection_values, Stream stream) const {

		for (int i = 0; i < tensor_dimension; ++i) {
			BC_ASSERT(dimension(0) == injection_values.data().dimension(0),
					"INVALID TENSOR INJECTION INCORRECT DIMENSIONS");
		}

		auto& injection = injection_values.data();

		if (expression_traits<Array>::requires_greedy_evaluation::value) {
			auto evaluated = greedy_evaluate(array, stream);

			BC::tensors::exprs::functions::conv2d_data_backwards(
					stream,
					injection,
					evaluated,
					padding, stride,
					alpha_mod, beta_mod);

				using vt = typename decltype(evaluated)::value_type;
				stream.template get_allocator_rebound<vt>().deallocate(evaluated.memptr(), evaluated.size());
		} else {

			BC::tensors::exprs::functions::conv2d_data_backwards(
					stream,
					injection,
					array,
					padding, stride,
					alpha_mod, beta_mod);
		}
	}
};



}
}
}


#endif /* FUNCTION_CONV2D_H_ */
