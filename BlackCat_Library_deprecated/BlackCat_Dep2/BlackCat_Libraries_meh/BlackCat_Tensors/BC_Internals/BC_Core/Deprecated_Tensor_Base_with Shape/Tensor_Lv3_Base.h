/*
 * Tensor_Lv3_Base.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef TENSOR_LV3_BASE_H_
#define TENSOR_LV3_BASE_H_

#include "Tensor_Lv2_Core.h"

namespace BC {

template<class T, class mid_deriv, class Mathlib, bool Utility_Function_Supported = false>
struct Tensor_Base_impl {
};


template<class scalar_type, class deriv, class MATHLIB>
struct Tensor_Base_impl<scalar_type, deriv, MATHLIB, true> {
/*
 *  Tensor_Base specialization (for primary tensors -- we enable these utility methods)
 */

	deriv& asDerived() {
		return static_cast<deriv&>(*this);
	}
	const deriv& convert() const {
		return static_cast<const deriv&>(*this);
	}

	void randomize(scalar_type lb, scalar_type ub) {
		MATHLIB::randomize(asDerived().data(), lb, ub, asDerived().size());
	}
	void fill(scalar_type value) {
		MATHLIB::fill(asDerived().data(), value, asDerived().size());
	}
	void zero() {
		MATHLIB::zero(asDerived().data(), asDerived().size());
	}
	void print() const {
		auto ranks = convert().getShape();
		MATHLIB::print(convert().array, ranks, convert().order(), 5);
	}

	Tensor_Base_impl() {
		if (asDerived().size() != 0)
		MATHLIB::initialize(asDerived().data(), asDerived().size());
	}

	~Tensor_Base_impl() {
		MATHLIB::destroy(static_cast<deriv&>(const_cast<Tensor_Base_impl&>(*this)).data());
	}
};



template<
	class scalar_type,

	template<class>
	class IDENTITY,
	class DERIVED,
	class MATHLIB,
	class SHAPE,
	class isParent>

struct Tensor_Base :

	public Tensor_Core<scalar_type, IDENTITY ,DERIVED, MATHLIB>,
	public SHAPE,
	public Tensor_Base_impl<scalar_type, DERIVED, MATHLIB, MTF::isPrimitive<scalar_type>::conditional>
{

	using primary_parent = Tensor_Core<scalar_type, IDENTITY ,DERIVED, MATHLIB>;
	using primary_parent::primary_parent;
	using primary_parent::operator=;

private:

		template<int>
		struct BANNED_METHOD { using type = void; BANNED_METHOD(int = 0) { throw std::invalid_argument("prohibited method call"); } };
		static constexpr bool UTILITY_FUNCTIONS_ENABLED = MTF::isPrimitive<scalar_type>::conditional;
		static constexpr bool UFE = UTILITY_FUNCTIONS_ENABLED;
		static constexpr bool INNER_DYNAMIC_SHAPE = SHAPE::inner_shape::isDynamic;
		static constexpr bool OUTER_DYNAMIC_SHAPE = SHAPE::outer_shape::isDynamic;
		static constexpr bool isChild = MTF::isFalse<isParent::conditional>::conditional;
		using inner_dims   = typename MTF::IF_ELSE<MTF::AND<UFE, INNER_DYNAMIC_SHAPE>::conditional,std::initializer_list<int>,BANNED_METHOD<0>>::type;
		using inner_ptr    = typename MTF::IF_ELSE<MTF::AND<UFE, INNER_DYNAMIC_SHAPE, isChild>::conditional,std::initializer_list<int>,BANNED_METHOD<1>>::type;
		using outer_dims   = typename MTF::IF_ELSE<MTF::AND<UFE, OUTER_DYNAMIC_SHAPE, isChild>::conditional, int*, BANNED_METHOD<2>>::type;
		using functor_type = typename primary_parent::functor_type;

public:


	functor_type array;
	template<class ... params> Tensor_Base(const params&... p) : array(p...) {}

	template< class... params, class inner, class outer> Tensor_Base(const Tensor_Shape<inner, outer>& s,  const params&... p) : SHAPE(s), array(p...) {}
	template<class ... params> Tensor_Base(inner_dims inner) : SHAPE(inner){}
	template<class ... params> Tensor_Base(inner_ptr inner) : SHAPE(inner) {}
	template<class ... params> Tensor_Base(inner_ptr inner, outer_dims outer) : SHAPE(inner, outer) {}

	template<class ... params> Tensor_Base(inner_dims inner, const params&... p) : SHAPE(inner), array(p...) {}
	template<class ... params> Tensor_Base(inner_dims inner, outer_dims outer, const params&... p) : SHAPE(inner, outer), array(p...) {}
	template<class ... params> Tensor_Base(outer_dims outer, const params&... p)                   : SHAPE(       outer), array(p...) {}

};

}


#endif /* TENSOR_LV3_BASE_H_ */
