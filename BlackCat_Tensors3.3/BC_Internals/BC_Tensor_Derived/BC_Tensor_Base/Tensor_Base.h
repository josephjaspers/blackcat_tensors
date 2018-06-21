
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
#include "Tensor_Initializer.h"
#include "Tensor_Shaping.h"
#include "Tensor_Functions.h"

namespace BC {

template<class derived>
class Tensor_Base :
		public Base::Tensor_Operations<derived>,
		public Base::Tensor_Functions<derived>,
		public Base::Tensor_Utility<derived>,
		public Base::Tensor_Initializer<derived>,
		public Base::Tensor_Shaping<derived>
{

protected:

	using self 			= Tensor_Base<derived>;
	using operations  	= Base::Tensor_Operations<derived>;
	using initializer 	= Base::Tensor_Initializer<derived>;
	using utility		= Base::Tensor_Utility<derived>;
	using shaping		= Base::Tensor_Shaping<derived>;

	using functor_type 	= _functor<derived>;
	using scalar_type	= _scalar<derived>;
	using mathlib_type 	= _mathlib<derived>;

	template<class> friend class Tensor_Base;

public:
	using operations::operator=;
	using shaping::operator[];
	using shaping::operator();

	operator const derived& () const { return static_cast<const derived&>(*this); }
	operator	   derived& () 		 { return static_cast< 		derived&>(*this); }


	template<class... params> explicit Tensor_Base(const params&... p) : initializer(p...) {}

	//move only defined for primary cores (this is to ensure slices/chunks/reshapes apply copies)
	using move_parameter = std::conditional_t<is_array_core<functor_type>(), derived&&, DISABLED>;
	Tensor_Base(move_parameter tensor) : initializer(std::move(tensor)) {}
	Tensor_Base(const Tensor_Base& 	tensor) : initializer(tensor) {}


	derived& operator =(move_parameter tensor) {
		auto tmp = this->internal();
		this->internal() = tensor.internal();
		tensor.internal() = tmp;

		return *this;
	}

	derived& operator =(const derived& tensor) {
		this->assert_same_size(tensor);
		mathlib_type::copy(this->internal(), tensor.internal(), this->size());
		return *this;
	}
	derived& operator =(scalar_type scalar) {
		this->fill(scalar);
		return *this;
	}
};

}

#endif /* TENSOR_BASE_H_ */

