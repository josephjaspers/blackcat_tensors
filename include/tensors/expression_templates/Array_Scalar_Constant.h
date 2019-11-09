/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_

#include "Expression_Template_Base.h"

namespace BC {
namespace tensors {
namespace exprs {

template<class Derived>
struct Scalar_Constant_Base : Shape<0>, Kernel_Array_Base<Derived> {

    static constexpr int tensor_iterator_dimension = 0;
    static constexpr int tensor_dimension = 0;

    using copy_constructible = std::false_type;
    using move_constructible = std::false_type;
    using copy_assignable    = std::false_type;
    using move_assignable    = std::false_type;
};

template<class Scalar, class SystemTag>
struct Scalar_Constant:
		Scalar_Constant_Base<Scalar_Constant<Scalar, SystemTag>> {

    using value_type = Scalar;
    using system_tag = SystemTag;
    using allocation_tag = BC::host_tag;
	using stack_allocated = std::true_type;
	using allocation_type = BC::host_tag;

    value_type scalar;

    BCINLINE Scalar_Constant(value_type scalar_) : scalar(scalar_) {}

    template<class... integers>
    BCINLINE auto operator()  (const integers&...) const { return scalar; }

    BCINLINE auto operator [] (int) const { return scalar; }
    BCINLINE const value_type* data() const { return &scalar; }
};


template<class SystemTag, class value_type>
auto make_scalar_constant(value_type scalar) {
    return Scalar_Constant<value_type, SystemTag>(scalar);
}


template<int Value, class Scalar, class SystemTag>
struct Constexpr_Scalar_Constant;

template<int Value, class Scalar>
struct Constexpr_Scalar_Constant<Value, Scalar, BC::host_tag>:
	Scalar_Constant_Base<Constexpr_Scalar_Constant<Value, Scalar, BC::host_tag>> {

	using value_type = Scalar;
    using system_tag = BC::host_tag;
    using allocation_tag = BC::host_tag;

    Scalar value = Scalar(Value);

    template<class... integers> BCINLINE auto operator()  (const integers&...) const { return Value; }
    template<class... integers> BCINLINE auto operator()  (const integers&...) 		 { return Value; }

    BCINLINE auto operator [] (int i ) const { return Value; }
    BCINLINE auto operator [] (int i )  	 { return Value; }

    BCHOT const Scalar* data() const { return &value; }
};


#ifdef __CUDACC__
template<int Value, class Scalar>
struct Constexpr_Scalar_Constant<Value, Scalar, BC::device_tag>:
	Scalar_Constant_Base<Constexpr_Scalar_Constant<Value, Scalar, BC::device_tag>> {

    using value_type = Scalar;
    using system_tag = BC::host_tag;
    using allocation_tag = BC::host_tag;

    const Scalar* value = cuda_constexpr_scalar_ptr();

    template<class... integers> BCINLINE auto operator()  (const integers&...) const { return Value; }
    template<class... integers> BCINLINE auto operator()  (const integers&...) 		 { return Value; }

    BCINLINE auto operator [] (int i ) const { return Value; }
    BCINLINE auto operator [] (int i )  	 { return Value; }

    BCHOT const Scalar* data() const { return value; }


  private:
	static const Scalar* cuda_constexpr_scalar_ptr() {

		static Scalar* scalar_constant_  = [](){
			Scalar tmp_val = Value;
			Scalar* scalar_constant_ = nullptr;
			BC_CUDA_ASSERT(cudaMalloc((void**)&scalar_constant_, sizeof(Scalar)));
			BC_CUDA_ASSERT(cudaMemcpy(scalar_constant_, &tmp_val, sizeof(Scalar), cudaMemcpyHostToDevice));
			return scalar_constant_;
		}();

		return scalar_constant_;
	}
};
#endif

template<class SystemTag, int Value, class Scalar>
auto make_constexpr_scalar() {
	return Constexpr_Scalar_Constant<Value, Scalar, SystemTag>();
}


} //ns BC
} //ns exprs
} //ns tensors




#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_ */
