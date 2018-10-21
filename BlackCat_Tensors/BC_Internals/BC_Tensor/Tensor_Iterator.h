/*
 * Tensor_STL_interface.h
 *
 *  Created on: Oct 18, 2018
 *      Author: joseph
 */

#ifndef TENSOR_STL_INTERFACE_H_
#define TENSOR_STL_INTERFACE_H_

#include "Tensor_Common.h"
#include "STL_esque/Iterator.h"
namespace BC {
namespace module {

	template<class derived>
	class Tensor_Iterator {

		derived& as_derived() {
			return static_cast<derived&>(*this);
		}
		const derived& as_derived() const {
			return static_cast<derived&>(*this);
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


	};

}
}



#endif /* TENSOR_STL_INTERFACE_H_ */
