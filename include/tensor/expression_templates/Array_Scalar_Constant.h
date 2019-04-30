/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_


#include "Array_Base.h"


namespace BC {
namespace exprs {


//identical to Array_Scalar, though the scalar is allocated on the stack opposed to heap
template<class Scalar, class SystemTag>
struct Scalar_Constant : Shape<0>, Array_Base<Scalar_Constant<Scalar, SystemTag>, 0>{

    using value_type = Scalar;
    using system_tag = SystemTag;

    static constexpr int ITERATOR = 0;
    static constexpr int DIMS     = 0;

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


template<int Value, class Scalar>
struct Constexpr_Scalar_Constant : Shape<0>, Array_Base<Constexpr_Scalar_Constant<Value, Scalar>, 0>{

    using value_type = Scalar;
    using system_tag = BC::host_tag;

    static constexpr int ITERATOR = 0;
    static constexpr int DIMS     = 0;

    Scalar value = Scalar(Value);

    template<class... integers> BCINLINE auto operator()  (const integers&...) const { return Value; }
    template<class... integers> BCINLINE auto operator()  (const integers&...) 		 { return Value; }

    BCINLINE auto operator [] (int i ) const { return Value; }
    BCINLINE auto operator [] (int i )  	 { return Value; }

    BCHOT const Scalar* memptr() const { return &value; }
};

template<int Value, class Scalar>
auto make_constexpr_scalar() {
	return Constexpr_Scalar_Constant<Value, Scalar>();
}

}
}



#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_ */
