/*
 * Tensor_Function_std.h
 *
 *  Created on: Oct 29, 2018
 *      Author: joseph
 */

#ifndef TENSOR_FUNCTION_STD_H_
#define TENSOR_FUNCTION_STD_H_

namespace BC {
namespace module {
namespace stl {

template<class derived>
class Tensor_std_functional {

    auto& as_derived() {
        return static_cast<derived&>(*this);
    }
    auto& as_derived() const {
        return static_cast<const derived&>(*this);
    }

public:

#define BC_TENSOR_FUNCTIONAL_STD_DEF(functor)     \
                                                \
                                                \
                                                \
                                                \
                                                \
                                                \

};
}
}
}



#endif /* TENSOR_FUNCTION_STD_H_ */
