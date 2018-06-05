
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

	using functor_type 	= _functor<derived>;
	using scalar_type	= _scalar<derived>;
	using mathlib_type 	= _mathlib<derived>;

	static constexpr int DIMS() { return dimension_of<derived>::value; }
	static constexpr int ITERATOR() { return functor_type::ITERATOR(); }

	template<class> friend class Tensor_Base;

public:
	using operations::operator=;

	operator const derived& () const { return static_cast<const derived&>(*this); }
	operator	   derived& () 		 { return static_cast< 		derived&>(*this); }


	template<class... params> explicit Tensor_Base(const params&... p) : initializer(p...) {}

	//move only defined for primary cores (this is to ensure slices/chunks/reshapes apply copies)
	using move_parameter = std::conditional_t<pCore_b<functor_type>, derived&&, DISABLED>;
	Tensor_Base(move_parameter tensor) : initializer(std::move(tensor)) {}
	Tensor_Base(const Tensor_Base& 	tensor) : initializer(tensor) {}


	derived& operator =(move_parameter tensor) {
		auto tmp = this->array_core;
		this->array_core = tensor.array_core;
		tensor.array_core = tmp;

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

	int dims() const { return this->array_core.dims(); }
	int size() const { return this->array_core.size(); }
	int rows() const { return this->array_core.rows(); }
	int cols() const { return this->array_core.cols(); }
	int outer_dimension() const { return this->array_core.outer_dimension(); }

	int ld1() const { return this->array_core.ld1(); }
	int ld2() const { return this->array_core.ld2(); }

	int dimension(int i)		const { return this->array_core.dimension(i); }
	void print_dimensions() 		const { this->array_core.print_dimensions();   }
	void print_leading_dimensions()	const { this->array_core.print_leading_dimensions(); }

	const auto inner_shape() const 			{ return this->array_core.inner_shape(); }
	const auto outer_shape() const 			{ return this->array_core.outer_shape(); }


};

}

#endif /* TENSOR_BASE_H_ */

