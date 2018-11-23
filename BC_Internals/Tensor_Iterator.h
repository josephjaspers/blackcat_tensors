/*
 * Tensor_STL_interface.h
 *
 *  Created on: Oct 18, 2018
 *      Author: joseph
 */

#ifndef TENSOR_STL_INTERFACE_H_
#define TENSOR_STL_INTERFACE_H_

#include "Tensor_Common.h"
#include "iterators/Multidimensional_Iterator.h"
#include "iterators/Coefficientwise_Iterator.h"

namespace BC {
namespace module {

    template<class derived>
    class Tensor_Iterator {

        derived& as_derived() {
            return static_cast<derived&>(*this);
        }
        const derived& as_derived() const {
            return static_cast<const derived&>(*this);
        }

    public:
        //------------------------multidimension_iterator------------------------//

        auto nd_begin() {
            return stl::forward_iterator_begin(as_derived());
        }
        auto nd_end() {
            return stl::forward_iterator_end(as_derived());
        }
        const auto nd_cbegin() const {
            return stl::forward_iterator_begin(as_derived());
        }
        const auto nd_cend() const {
            return stl::forward_iterator_end(as_derived());
        }
        auto nd_rbegin() {
            return stl::reverse_iterator_begin(as_derived());
        }
        auto nd_rend() {
            return stl::reverse_iterator_end(as_derived());
        }
        const auto nd_crbegin() const {
            return stl::reverse_iterator_begin(as_derived());
        }
        const auto nd_crend() const {
            return stl::reverse_iterator_end(as_derived());
        }
        //----------const versions----------//
        auto nd_begin() const {
            return stl::forward_iterator_begin(as_derived());
        }
        auto nd_end() const {
            return stl::forward_iterator_end(as_derived());
        }
        auto nd_rbegin() const {
            return stl::reverse_iterator_begin(as_derived());
        }
        auto nd_rend() const {
            return stl::reverse_iterator_end(as_derived());
        }
        //------------------------elementwise_iterator------------------------//
        auto begin() {
            return stl::forward_cwise_iterator_begin(as_derived().internal());
        }
        auto end() {
            return stl::forward_cwise_iterator_end(as_derived().internal());
        }
        const auto cbegin() const {
            return stl::forward_cwise_iterator_begin(as_derived().internal());
        }
        const auto cend() const {
            return stl::forward_cwise_iterator_end(as_derived().internal());
        }
        auto rbegin() {
            return stl::reverse_cwise_iterator_begin(as_derived().internal());
        }
        auto rend() {
            return stl::reverse_cwise_iterator_end(as_derived().internal());
        }
        const auto crbegin() const {
            return stl::reverse_cwise_iterator_begin(as_derived().internal());
        }
        const auto crend() const {
            return stl::reverse_cwise_iterator_end(as_derived().internal());
        }
        //const versions
        auto begin() const {
            return stl::forward_cwise_iterator_begin(as_derived().internal());
        }
        auto end() const {
            return stl::forward_cwise_iterator_end(as_derived().internal());
        }
        auto rbegin() const {
            return stl::reverse_cwise_iterator_begin(as_derived().internal());
        }
        auto rend() const {
            return stl::reverse_cwise_iterator_end(as_derived().internal());
        }



        //----------------------iterator wrappers---------------------------//

#define BC_TENSOR_ITERATOR_DEF(iterator_name, begin_func, end_func)\
    template<class der_t>                                        \
    struct iterator_name {                                       \
                                                                 \
        der_t& tensor;                                           \
                                                                 \
        using begin_t = decltype(tensor.begin_func ());          \
        using end_t = decltype(tensor.end_func ());              \
                                                                 \
        begin_t _begin = tensor.begin_func();                    \
        end_t _end = tensor.end_func();                          \
                                                                 \
        iterator_name(der_t& tensor_) :                          \
                tensor(tensor_) {                                \
        }                                                        \
                                                                 \
        iterator_name(der_t& tensor_, int start) :               \
                tensor(tensor_) {                                \
                                                                 \
            _begin += start;                                     \
        }                                                        \
        iterator_name(der_t& tensor_, int start, int end) :      \
                tensor(tensor_) {                                \
            _begin += start;                                     \
            _end = end;                                          \
        }                                                        \
        auto begin() {                                           \
            return _begin;                                       \
        }                                                        \
        const begin_t& cbegin() const {                          \
            return _begin;                                       \
        }                                                        \
        const end_t& end() const {                               \
            return _end;                                         \
        }                                                        \
                                                                 \
    };                                                           \
                                                                 \
 template<class der_t, class... args>                            \
 static auto make_##iterator_name (der_t& p_derived, args... params) {    \
       return iterator_name<der_t>(p_derived, params...);                 \
 }                                                                        \

BC_TENSOR_ITERATOR_DEF(ND_ForwardIterator, nd_begin, nd_end)
BC_TENSOR_ITERATOR_DEF(ND_ReverseIterator, nd_rbegin, nd_rend)
BC_TENSOR_ITERATOR_DEF(CW_ForwardIterator, begin, end)
BC_TENSOR_ITERATOR_DEF(CW_ReverseIterator, rbegin, rend)

template<class... params> auto iter(params ... ps) {
    return make_CW_ForwardIterator(as_derived(), ps...);
}
template<class... params> auto reverse_iter(params ... ps) {
    return make_CW_ReverseIterator(as_derived(), ps...);
}
template<class... params> auto iter(params ... ps) const {
    return make_CW_ForwardIterator(as_derived(), ps...);
}
template<class... params> auto reverse_iter(params ... ps) const {
    return make_CW_ReverseIterator(as_derived(), ps...);
}
template<class... params> auto nd_iter(params ... ps) {
    return make_ND_ForwardIterator(as_derived(), ps...);
}
template<class... params> auto nd_reverse_iter(params ... ps) {
    return make_ND_ReverseIterator(as_derived(), ps...);
}
template<class... params> auto nd_iter(params ... ps) const {
    return make_ND_ForwardIterator(as_derived(), ps...);
}
template<class... params> auto nd_reverse_iter(params ... ps) const {
    return make_ND_ReverseIterator(as_derived(), ps...);
}
};
}
}

#endif /* TENSOR_STL_INTERFACE_H_ */
