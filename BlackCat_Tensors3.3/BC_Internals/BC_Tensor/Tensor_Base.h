
/*
 * Tensor_Base.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef TENSOR_BASE_H_
#define TENSOR_BASE_H_

#include "Tensor_Operations.h"
#include "Tensor_Utility.h"
#include "Tensor_Shaping.h"
#include "Tensor_Functions.h"

#include "Expression_Templates/Array.h"
#include "Expression_Templates/Array_Scalar.h"
#include "Expression_Templates/Array_Slice.h"
#include "Expression_Templates/Array_Slice_Complex.h"
#include "Expression_Templates/Array_Chunk.h"
#include "Expression_Templates/Array_Reshape.h"
namespace BC {

template<class internal_t>
class Tensor_Base :

		public Base::Tensor_Operations<Tensor_Base<internal_t>>,
		public Base::Tensor_Functions<Tensor_Base<internal_t>>,
		public Base::Tensor_Utility<Tensor_Base<internal_t>>,
		public Base::Tensor_Shaping<Tensor_Base<internal_t>>,
		public internal_t {

protected:

	using self 			= Tensor_Base<internal_t>;
	using operations  	= Base::Tensor_Operations<Tensor_Base<internal_t>>;
	using utility		= Base::Tensor_Utility<Tensor_Base<internal_t>>;
	using shaping		= Base::Tensor_Shaping<Tensor_Base<internal_t>>;

	using scalar_type	= _scalar<internal_t>;
	using mathlib_type 	= _mathlib<internal_t>;

	template<class> friend class Tensor_Base;

public:

	using operations::operator=;
	using shaping::operator[];
	using shaping::operator();
	using internal_t::internal_t;

	using internal_t::DIMS; //required
	using scalar_t = typename internal_t::scalar_t;

	using move_parameter = std::conditional_t<is_array_core<internal_t>(), self&&, BC::DISABLED<0>>;
	using copy_parameter = std::conditional_t<is_array_core<internal_t>(), const self&, BC::DISABLED<1>>;


	Tensor_Base() = default;
	Tensor_Base(copy_parameter tensor) : internal_t(tensor.inner_shape()) {
		mathlib_type::copy(this->internal(), tensor.internal(), this->size());
	}
	Tensor_Base(move_parameter tensor) : internal_t(tensor.inner_shape()) {
		std::swap(this->array, tensor.array);
		this->swap_shape(tensor);
	}
	template<class U>
	Tensor_Base(const Tensor_Base<U>&  tensor) : internal_t(tensor.inner_shape()) {
		operations::operator=(tensor);
	}
	Tensor_Base(const internal_t&  param) : internal_t(param) {}
	Tensor_Base( 	  internal_t&& param) : internal_t(param) {}

	Tensor_Base& operator =(copy_parameter tensor) {
		this->assert_valid(tensor);
		this->copy_shape(tensor);//this is internal_t from shape
		mathlib_type::copy(this->internal(), tensor.internal(), this->size());
		return *this;
	}

	Tensor_Base& operator =(move_parameter tensor) {
		this->swap_shape(tensor);
		this->swap_array(tensor);
		return *this;
	}

	Tensor_Base(scalar_t scalar) {
		static_assert(DIMS() == 0, "SCALAR_INITIALIZATION ONLY AVAILABLE TO SCALARS");
		this->fill(scalar);
	}
	Tensor_Base& operator =(scalar_t scalar) {
		this->fill(scalar);
		return *this;
	}
	~Tensor_Base() {
		if (is_array_core<internal_t>()) {
			this->destroy();
		}
	}
	 const internal_t& internal() const { return static_cast<const internal_t&>(*this); }
	 	   internal_t& internal() 	  	 { return static_cast<	    internal_t&>(*this); }
};

}

#endif /* TENSOR_BASE_H_ */
