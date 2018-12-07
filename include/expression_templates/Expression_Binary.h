/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_BINARY_POINTWISE_SAME_H_
#define EXPRESSION_BINARY_POINTWISE_SAME_H_

#include "Expression_Base.h"

namespace BC {
namespace et     {
template<class lv, class rv, class operation>
struct Binary_Expression : public Expression_Base<Binary_Expression<lv, rv, operation>>, public operation {

    using scalar_t = decltype(std::declval<operation>()(std::declval<typename lv::scalar_t&>(), std::declval<typename lv::scalar_t&>()));
    using allocator_t = typename lv::allocator_t;
    using system_tag  = typename lv::system_tag;

    lv left;
    rv right;

    template<class L, class R> __BCinline__ const auto oper(const L& l, const R& r) const { return static_cast<const operation&>(*this)(l,r); }
    template<class L, class R> __BCinline__       auto oper(const L& l, const R& r)        { return static_cast<      operation&>(*this)(l,r); }

    __BCinline__ static constexpr int DIMS() { return lv::DIMS() > rv::DIMS() ?  lv::DIMS() : rv::DIMS(); }
    __BCinline__ static constexpr int ITERATOR() {
        //if dimension mismatch choose the max dimension as iterator, else choose the max iterator
        return lv::DIMS() != rv::DIMS() ? DIMS() : MTF::max(lv::ITERATOR(), rv::ITERATOR());
    }

    template<class... args> __BChot__
    Binary_Expression(lv l, rv r, const args&... args_) :  operation(args_...), left(l), right(r) {}

    __BCinline__  auto  operator [](int index) const { return oper(left[index], right[index]); }
    template<class... integers> __BCinline__  auto  operator ()(integers... ints) const { return oper(left(ints...), right(ints...)); }

    __BCinline__ const auto& shape() const { return dominant_type<lv, rv>::shape(left, right); }
    __BCinline__ int size() const { return shape().size(); }
    __BCinline__ int rows() const { return shape().rows(); }
    __BCinline__ int cols() const { return shape().cols(); }
    __BCinline__ int dimension(int i) const { return shape().dimension(i); }
    __BCinline__ int block_dimension(int i) const { return shape().block_dimension(i); }
    __BCinline__ const auto inner_shape() const { return shape().inner_shape(); }
    __BCinline__ const auto block_shape() const { return shape().block_shape(); }
};

template<class op, class lv, class rv>
auto make_bin_expr(lv left, rv right) {
	return Binary_Expression<lv, rv, op>(left, right);
}

}
}


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */

//
//private:
//    template<class l, class r>
//    auto make_binexpr(l l_, r r_, const operation& op) {
//        return Binary_Expression<l, r, operation>(l_, r_, op);
//    }
//
//    const operation& as_op() {
//        return static_cast<const operation&>(*this);
//    }
//public:
//
//    __BCinline__ const auto _slice(int i) const {
//        return make_binexpr(left._slice(i), right._slice(i), this->as_op());
//    }
//    __BCinline__ const auto _scalar(int i) const {
//        return make_binexpr(left._scalar(i), right._scalar(i), this->as_op());
//    }
//    __BCinline__ const auto _slice_range(int from, int to) {
//        return make_binexpr(left._slice_range(from, to), right._slice_range(from, to),  this->as_op());
//    }

