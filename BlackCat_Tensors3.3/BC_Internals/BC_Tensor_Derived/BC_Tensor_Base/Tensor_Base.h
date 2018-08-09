
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

template<class derived>
class Tensor_Base :

		public Base::Tensor_Operations<Tensor_Base<derived>>,
		public Base::Tensor_Functions<Tensor_Base<derived>>,
		public Base::Tensor_Utility<Tensor_Base<derived>>,
		public Base::Tensor_Shaping<Tensor_Base<derived>>,
		public derived {

protected:

	using self 			= Tensor_Base<derived>;
	using operations  	= Base::Tensor_Operations<Tensor_Base<derived>>;
	using utility		= Base::Tensor_Utility<Tensor_Base<derived>>;
	using shaping		= Base::Tensor_Shaping<Tensor_Base<derived>>;

	using scalar_type	= _scalar<derived>;
	using mathlib_type 	= _mathlib<derived>;

	template<class> friend class Tensor_Base;

public:

	using operations::operator=;
	using shaping::operator[];
	using shaping::operator();
	using derived::derived;

	using derived::DIMS;//This is important -- if removed ambiguous calls from the super classes
	using scalar_t = typename derived::scalar_t;


	Tensor_Base() = default;
	Tensor_Base(const Tensor_Base&  tensor) : derived(tensor.internal()) {}
	Tensor_Base(      Tensor_Base&& tensor) : derived(std::move(tensor.internal())) {}
	Tensor_Base(const derived& param) : derived(param) {}
	Tensor_Base( derived&& param) : derived(param) {}

	//	template<class... params> explicit Tensor_Base(const params&...  ps) : derived(ps...) {}
//	template<class... params> explicit Tensor_Base(		 params&&... ps) : derived(ps...) {}
	//copy move (only availble to "primary" array types (non-expressions)
	using move_parameter = std::conditional_t<is_array_core<derived>(), self&&, BC::DISABLED<0>>;
	using copy_parameter = std::conditional_t<is_array_core<derived>(), const self&, BC::DISABLED<1>>;

	Tensor_Base& operator =(copy_parameter tensor) {
		this->assert_valid(tensor);
		this->copy_shape(tensor);//this is derived from shape
		mathlib_type::copy(this->internal(), tensor.internal(), this->size());
		return *this;
	}
//
	Tensor_Base& operator =(move_parameter tensor) {
		this->swap_shape(tensor);
		std::swap(this->array, tensor.array);
		return *this;
	}

//-------------------------------------------FILL CONSTRUCTORS/=OPERATOR--------------------------------//
	Tensor_Base(scalar_t scalar) {
		static_assert(DIMS() == 0, "SCALAR_INITIALIZATION ONLY AVAILABLE TO SCALARS");
		this->fill(scalar);
	}
	Tensor_Base& operator =(scalar_t scalar) {
		this->fill(scalar);
		return *this;
	}

	template<class U>
	Tensor_Base(const Tensor_Base<U>&  tensor) : derived(tensor.inner_shape()) {
		operations::operator=(tensor);
	}


	~Tensor_Base() {


		if (is_array_core<derived>()) {
//		std::cout << " destructor " << std::endl;
//		this->internal().destroy();
		}
	}

	 const derived& internal() const { return static_cast<const derived&>(*this); }
	 	   derived& internal() 	  	 { return static_cast<	    derived&>(*this); }
};

}

#endif /* TENSOR_BASE_H_ */



////move only defined for primary cores (this is to ensure slices/chunks/reshapes apply copies)
//using move_parameter = std::conditional_t<is_array_core<derived>(), self&&, BC::DISABLED<0>>;
//using copy_parameter = std::conditional_t<is_array_core<derived>(), const self&, BC::DISABLED<1>>;
