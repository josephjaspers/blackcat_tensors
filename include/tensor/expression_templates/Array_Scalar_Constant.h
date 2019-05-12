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
#include "Shape.h"

namespace BC {
namespace exprs {


//identical to Array_Scalar, though the scalar is allocated on the stack opposed to heap
template<class Scalar, class SystemTag>
struct Scalar_Constant : Shape<0>, Array_Base<Scalar_Constant<Scalar, SystemTag>, 0>, BC::exprs::BC_Stack_Allocated {

    using value_type = Scalar;
    using system_tag = SystemTag;
    using allocation_tag = BC::host_tag;

    static constexpr int ITERATOR = 0;
    static constexpr int DIMS     = 0;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = true;
    static constexpr bool move_assignable    = true;


    value_type scalar;

    BCINLINE Scalar_Constant(value_type scalar_) : scalar(scalar_) {}

    template<class... integers> BCINLINE auto operator()  (const integers&...) const { return scalar; }
    template<class... integers> BCINLINE auto operator()  (const integers&...) 		 { return scalar; }

    BCINLINE auto operator [] (int i ) const { return scalar; }
    BCINLINE auto operator [] (int i )  	 { return scalar; }

    BCINLINE const value_type* memptr() const { return &scalar; }
};


template<class SystemTag, class value_type>
auto make_scalar_constant(value_type scalar) {
    return Scalar_Constant<value_type, SystemTag>(scalar);
}


template<int Value, class Scalar, class SystemTag>
struct Constexpr_Scalar_Constant;

template<int Value, class Scalar>
struct Constexpr_Scalar_Constant<Value, Scalar, BC::host_tag>
: Shape<0>,
  Array_Base<Constexpr_Scalar_Constant<Value, Scalar, BC::host_tag>, 0>{

    using value_type = Scalar;
    using system_tag = BC::host_tag;
    using allocation_tag = BC::host_tag;

    static constexpr int ITERATOR = 0;
    static constexpr int DIMS     = 0;

    static constexpr bool copy_constructible = false;
    static constexpr bool move_constructible = false;
    static constexpr bool copy_assignable    = false;
    static constexpr bool move_assignable    = false;

    Scalar value = Scalar(Value);

    template<class... integers> BCINLINE auto operator()  (const integers&...) const { return Value; }
    template<class... integers> BCINLINE auto operator()  (const integers&...) 		 { return Value; }

    BCINLINE auto operator [] (int i ) const { return Value; }
    BCINLINE auto operator [] (int i )  	 { return Value; }

    BCHOT const Scalar* memptr() const { return &value; }
};


#ifdef __CUDACC__
template<int Value, class Scalar>
struct Constexpr_Scalar_Constant<Value, Scalar, BC::device_tag>
: Shape<0>,
  Array_Base<Constexpr_Scalar_Constant<Value, Scalar, BC::device_tag>, 0>{

    using value_type = Scalar;
    using system_tag = BC::host_tag;
    using allocation_tag = BC::host_tag;

    static constexpr int ITERATOR = 0;
    static constexpr int DIMS     = 0;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = true;
    static constexpr bool move_assignable    = true;

    const Scalar* value = cuda_constexpr_scalar_ptr();

    template<class... integers> BCINLINE auto operator()  (const integers&...) const { return Value; }
    template<class... integers> BCINLINE auto operator()  (const integers&...) 		 { return Value; }

    BCINLINE auto operator [] (int i ) const { return Value; }
    BCINLINE auto operator [] (int i )  	 { return Value; }

    BCHOT const Scalar* memptr() const { return value; }


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

}
}



#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_ */
