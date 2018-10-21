/*
 * Tensor_STL_interface.h
 *
 *  Created on: Oct 18, 2018
 *      Author: joseph
 */

#ifndef TENSOR_STL_INTERFACE_H_
#define TENSOR_STL_INTERFACE_H_

#include "Tensor_Common.h"
#include "STL_Style_Iterators/Iterator.h"
#include "STL_Style_Iterators/Coefficientwise_Iterator.h"

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

		//element access ----------------------
		auto front() {
			return this->as_derived().slice(0);
		}
		auto back() {
			auto last = this->as_derived().outer_dimension();
			return this->as_derived().slice(last-1);
		}
		auto front() const {
			return this->as_derived().slice(0);
		}
		auto back() const {
			auto last = this->as_derived().outer_dimension();
			return this->as_derived().slice(last-1);
		}
		auto* data() {
			return as_derived().memptr();
		}
		auto* data() const {
			return as_derived().memptr();
		}

	public:
		//------------------------multidimension_iterator------------------------//

		auto begin() {
			return stl::forward_iterator_begin(as_derived());
		}
		auto end() {
			return stl::forward_iterator_end(as_derived());
		}
		const auto cbegin() const {
			return stl::forward_iterator_begin(as_derived());
		}
		const auto cend() const {
			return stl::forward_iterator_end(as_derived());
		}
		auto rbegin() {
			return stl::reverse_iterator_begin(as_derived());
		}
		auto rend() {
			return stl::reverse_iterator_end(as_derived());
		}
		const auto crbegin() const {
			return stl::reverse_iterator_begin(as_derived());
		}
		const auto crend() const {
			return stl::reverse_iterator_end(as_derived());
		}
		//------------------------elementwise_iterator------------------------//
		auto cwise_begin() {
			return stl::forward_cwise_iterator_begin(as_derived());
		}
		auto cwise_end() {
			return stl::forward_cwise_iterator_end(as_derived());
		}
		const auto cwise_cbegin() const {
			return stl::forward_cwise_iterator_begin(as_derived());
		}
		const auto cwise_cend() const {
			return stl::forward_cwise_iterator_end(as_derived());
		}
		auto cwise_rbegin() {
			return stl::reverse_cwise_iterator_begin(as_derived());
		}
		auto cwise_rend() {
			return stl::reverse_cwise_iterator_end(as_derived());
		}
		const auto cwise_crbegin() const {
			return stl::reverse_cwise_iterator_begin(as_derived());
		}
		const auto cwise_crend() const {
			return stl::reverse_cwise_iterator_end(as_derived());
		}


		//----------------------iterator wrappers---------------------------//

		struct _forward_iterator {

			derived& tensor;

			using begin_t = decltype(tensor.begin());
			using end_t   = decltype(tensor.end());

			begin_t _begin = tensor.begin();
			end_t   _end = tensor.end();

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

			using begin_t = decltype(tensor.rbegin());
			using end_t   = decltype(tensor.rend());

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


		template<class...params> auto iterator(params... ps) {
			return _forward_iterator(as_derived(), ps...);
		}
		template<class... params> auto reverse_iterator(params... ps) {
			return _reverse_iterator(as_derived(), ps...);
		}




	struct _cwise_forward_iterator {

		derived& tensor;

		using begin_t = decltype(tensor.cwise_begin());
		using end_t   = decltype(tensor.cwise_end());

		begin_t _begin = tensor.cwise_begin();
		end_t   _end = tensor.cwise_end();

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

		using begin_t = decltype(tensor.cwise_rbegin());
		using end_t   = decltype(tensor.cwise_rend());

		begin_t _begin = tensor.cwise_rbegin();
		end_t   _end = tensor.cwise_rend();

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

	template<class...params> auto cwise_iterator(params... ps) {
		return _cwise_forward_iterator(as_derived(), ps...);
	}
	template<class... params> auto cwise_reverse_iterator(params... ps) {
		return _cwise_reverse_iterator(as_derived(), ps...);
	}




};
}
}



#endif /* TENSOR_STL_INTERFACE_H_ */
