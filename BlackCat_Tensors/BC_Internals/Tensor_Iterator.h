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

        struct _forward_iterator {

            derived& tensor;

            using begin_t = decltype(tensor.nd_begin());
            using end_t   = decltype(tensor.nd_end());

            begin_t _begin = tensor.nd_begin();
            end_t   _end = tensor.nd_end();

            _forward_iterator(derived& tensor_)
                : tensor(tensor_) {}

            _forward_iterator(derived& tensor_, int start)
                : tensor(tensor_) {

                _begin += start;
            }
            _forward_iterator(derived& tensor_, int start, int end)
                : tensor(tensor_) {
                _begin += start;
                _end = end;
            }

            auto begin() {
                return _begin;
            }
            const begin_t& cbegin() const {
                return _begin;
            }
            const end_t& end() const {
                return _end;
            }

        };
        struct _reverse_iterator {

            derived& tensor;

            using begin_t = decltype(tensor.nd_rbegin());
            using end_t   = decltype(tensor.nd_rend());

            begin_t _begin = tensor.rbegin();
            end_t   _end = tensor.rend();

            _reverse_iterator(derived& tensor_)
                : tensor(tensor_) {}

            _reverse_iterator(derived& tensor_, int lower_index)
                : tensor(tensor_) {

                _end += lower_index;
            }
            _reverse_iterator(derived& tensor_, int lower, int higher)
                : tensor(tensor_) {
                _begin -= higher;
                _end = lower;
            }



            auto begin() {
                return _begin;
            }
            const begin_t& cbegin() const {
                return _begin;
            }
            const end_t& end() const {
                return _end;
            }

        };


        template<class...params> auto nd_iter(params... ps) {
            return _forward_iterator(as_derived(), ps...);
        }
        template<class... params> auto nd_reverse_iter(params... ps) {
            return _reverse_iterator(as_derived(), ps...);
        }




    struct _cwise_forward_iterator {

        derived& tensor;

        using begin_t = decltype(tensor.begin());
        using end_t   = decltype(tensor.end());

        begin_t _begin = tensor.begin();
        end_t   _end = tensor.end();

        _cwise_forward_iterator(derived& tensor_)
            : tensor(tensor_) {}

        _cwise_forward_iterator(derived& tensor_, int start)
            : tensor(tensor_) {

            _begin += start;
        }
        _cwise_forward_iterator(derived& tensor_, int start, int end)
            : tensor(tensor_) {
            _begin += start;
            _end = end;
        }

        auto begin() {
            return _begin;
        }
        const begin_t& cbegin() const {
            return _begin;
        }
        const end_t& end() const {
            return _end;
        }

    };
    struct _cwise_reverse_iterator {

        derived& tensor;

        using begin_t = decltype(tensor.rbegin());
        using end_t   = decltype(tensor.rend());

        begin_t _begin = tensor.rbegin();
        end_t   _end = tensor.rend();

        _cwise_reverse_iterator(derived& tensor_)
            : tensor(tensor_) {}

        _cwise_reverse_iterator(derived& tensor_, int lower_index)
            : tensor(tensor_) {

            _end += lower_index;
        }
        _cwise_reverse_iterator(derived& tensor_, int lower, int higher)
            : tensor(tensor_) {
            _begin -= higher;
            _end = lower;
        }

        auto begin() {
            return _begin;
        }
        const begin_t& cbegin() const {
            return _begin;
        }
        const end_t& end() const {
            return _end;
        }

    };

    template<class...params> auto iter(params... ps) {
        return _cwise_forward_iterator(as_derived(), ps...);
    }
    template<class... params> auto reverse_iter(params... ps) {
        return _cwise_reverse_iterator(as_derived(), ps...);
    }




};
}
}



#endif /* TENSOR_STL_INTERFACE_H_ */
