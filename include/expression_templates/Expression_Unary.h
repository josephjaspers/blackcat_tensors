/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_UNARY_POINTWISE_CU_
#define EXPRESSION_UNARY_POINTWISE_CU_

#include "Expression_Base.h"

namespace BC {
namespace et     {
template<class value, class operation>
struct Unary_Expression : public Expression_Base<Unary_Expression<value, operation>>, public operation {

    using scalar_t  = decltype(std::declval<operation>()(std::declval<typename value::scalar_t>()));
    using system_tag  = typename value::system_tag;
    using allocator_t = allocator::implementation<system_tag, scalar_t>;
    using utility_t	  = utility::implementation<system_tag>;

    __BCinline__ static constexpr int DIMS() { return value::DIMS(); }
    __BCinline__ static constexpr int ITERATOR() { return value::ITERATOR(); }

    value array;


    template<class... args> __BCinline__
    Unary_Expression(value v, const args&... args_) : operation(args_...) , array(v) {}

    __BCinline__ auto operator [](int index) const {
        return static_cast<const operation&>(*this)(array[index]);
    }
    template<class... integers>__BCinline__
    auto operator ()(integers... index) const {
        return static_cast<const operation&>(*this)(array(index...));
    }

    __BCinline__  const auto inner_shape() const { return array.inner_shape(); }
    __BCinline__  const auto block_shape() const { return array.block_shape(); }
    __BCinline__ int size() const { return array.size(); }
    __BCinline__ int rows() const { return array.rows(); }
    __BCinline__ int cols() const { return array.cols(); }
    __BCinline__ int dimension(int i) const { return array.dimension(i); }
    __BCinline__ int block_dimension(int i) const { return array.block_dimension(i); }
};

template<class op, class expr>
auto make_un_expr(expr e) {
	return Unary_Expression<expr, op>(e);
}

}
}
#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */



//__BCinline__ const auto _slice(int i) const {
//    using slice_t = decltype(array._slice(i));
//    return Unary_Expression<slice_t, operation>(array._slice(i), static_cast<const operation&>(*this));
//}
//__BCinline__ const auto _scalar(int i) const {
//    using scalar_t = decltype(array._scalar(i));
//    return Unary_Expression<scalar_t, operation>(array._scalar(i),  static_cast<const operation&>(*this));
//}
//__BCinline__ const auto _slice_range(int from, int to) const {
//    using scalar_t = decltype(array._slice_range(from, to));
//    return Unary_Expression<scalar_t, operation>(array._slice_range(from,to),  static_cast<const operation&>(*this));
//}
//
