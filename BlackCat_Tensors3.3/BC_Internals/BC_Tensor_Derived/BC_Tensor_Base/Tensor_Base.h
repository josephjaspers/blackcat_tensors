
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
		public Base::Tensor_Operations<Tensor_Base<derived>>,
		public Base::Tensor_Functions<Tensor_Base<derived>>,
		public Base::Tensor_Utility<Tensor_Base<derived>>,
		public Base::Tensor_Initializer<Tensor_Base<derived>>,
		public Base::Tensor_Shaping<Tensor_Base<derived>>
{

protected:

	using self 			= Tensor_Base<derived>;
	using operations  	= Base::Tensor_Operations<Tensor_Base<derived>>;
	using initializer 	= Base::Tensor_Initializer<Tensor_Base<derived>>;
	using utility		= Base::Tensor_Utility<Tensor_Base<derived>>;
	using shaping		= Base::Tensor_Shaping<Tensor_Base<derived>>;

	using functor_type 	= _functor<derived>;
	using scalar_type	= _scalar<derived>;
	using mathlib_type 	= _mathlib<derived>;

	template<class> friend class Tensor_Base;

public:

	using operations::operator=;
	using shaping::operator[];
	using shaping::operator();

	using initializer::DIMS;//This is important -- if removed ambiguous calls from the super classes
	using scalar_t = typename derived::scalar_t;

//	operator const derived& () const { return static_cast<const derived&>(*this); }
//	operator	   derived& () 		 { return static_cast< 		derived&>(*this); }

	template<class T, class... params> explicit Tensor_Base(const T& p1, const params&...  ps) : initializer(p1, ps...) {}
	template<class T, class... params> explicit Tensor_Base(T&& p1,	 params&&... ps) : initializer(p1, ps...) {}

	//move only defined for primary cores (this is to ensure slices/chunks/reshapes apply copies)
	using move_parameter = std::conditional_t<is_array_core<functor_type>(), derived&&, BC::DISABLED<0>>;
	using copy_parameter = std::conditional_t<is_array_core<functor_type>(), const derived&, BC::DISABLED<1>>;


	Tensor_Base() : initializer(Shape<DIMS()>()) {}
	Tensor_Base(move_parameter tensor) : initializer(std::move(tensor)) {}
	Tensor_Base(copy_parameter tensor) : initializer(tensor) {}

	Tensor_Base(scalar_t scalar) {
		static_assert(DIMS() == 0, "SCALAR_INITIALIZATION ONLY AVAILABLE TO SCALARS");
		this->fill(scalar);
	}

	//copy move (only availble to "primary" array types (non-expressions)
	derived& operator =(move_parameter tensor) {
		auto tmp = this->internal();
		this->internal() = tensor.internal();
		tensor.internal() = tmp;
		return *this;
	}

	derived& operator =(const copy_parameter& tensor) {
		this->assert_valid(tensor);
		mathlib_type::copy(this->internal(), tensor.internal(), this->size());
		return *this;
	}
	//"fill"
	derived& operator =(scalar_type scalar) {
		this->fill(scalar);
		return *this;
	}
};

}

#endif /* TENSOR_BASE_H_ */

