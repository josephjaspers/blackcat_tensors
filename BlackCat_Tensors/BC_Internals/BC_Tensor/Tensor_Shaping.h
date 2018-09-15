/*
 * Tensor_Shaping.h
 *
 *  Created on: May 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SHAPING_H_
#define TENSOR_SHAPING_H_

#include "Tensor_Common.h"

namespace BC {
namespace module {

template<class derived>
struct Tensor_Shaping {

	static constexpr int DIMS() { return derived::DIMS(); }

private:

	const auto& as_derived() const { return static_cast<const derived&>(*this); }
		  auto& as_derived()       { return static_cast<	  derived&>(*this); }

public:

	const auto operator [] (int i) const { return slice(i); }
		  auto operator [] (int i) 		 { return slice(i); }

	struct range { int from, to; };

	const auto operator [] (range r) const { return slice(r.from, r.to); }
		  auto operator [] (range r) 	   { return slice(r.from, r.to); }

	const auto scalar(int i) const { return make_tensor(as_derived()._scalar(i)); }
		  auto scalar(int i) 	   { return make_tensor(as_derived()._scalar(i)); }

	const auto slice(int i) const  { return make_tensor(as_derived()._slice(i)); }
		  auto slice(int i) 	   { return make_tensor(as_derived()._slice(i)); }

	const auto slice(int from, int to) const  { return make_tensor(as_derived()._slice_range(from, to)); }
		  auto slice(int from, int to) 	   	  { return make_tensor(as_derived()._slice_range(from, to)); }


	const auto col(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}
	auto col(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}
	const auto row(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return make_tensor(as_derived()._row(i));
	}
	auto row(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return make_tensor(as_derived()._row(i));
	}


	const auto operator() (int i) const { return scalar(i); }
		  auto operator() (int i) 	    { return scalar(i); }

	const auto& operator() () const { return *this; }
		  auto& operator() () 	    { return *this; }

	template<class... integers> const auto operator() (int i, integers... ints) const  {
		static_assert(MTF::seq_of<int, integers...>, "operator()(integers...) -> PARAMS MUST BE INTEGER LIST");
		return (*this)[i](ints...);
	}

	template<class... integers> 	  auto operator() (int i, integers... ints) {
		static_assert(MTF::seq_of<int, integers...>, "operator()(integers...) -> PARAMS MUST BE INTEGER LIST");
		return (*this)[i](ints...);
	}

};

}

template<class T>
const auto reshape(const Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		return make_tensor(tensor._reshape(integers...));
	};
}
template<class T>
auto reshape(Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		return make_tensor(tensor._reshape(integers...));
	};
}

template<class T, class... integers>// class enabler = std::enable_if_t<MTF::seq_of<int, integers...>>>
const auto chunk(const Tensor_Base<T>& tensor, integers... ints) {
	return [&](auto... shape_indicies) {
		return make_tensor(tensor._chunk(make_array(ints...), make_array(shape_indicies...)));
	};
}

template<class T, class... integers>//, class enabler = std::enable_if_t<MTF::seq_of<int, integers...>>>
 auto chunk( Tensor_Base<T>& tensor, integers... ints) {
	return [&](auto... shape_indicies) {
		return make_tensor(tensor._chunk(make_array(ints...), make_array(shape_indicies...)));
	};
}

}
#endif /* TENSOR_SHAPING_H_ */
