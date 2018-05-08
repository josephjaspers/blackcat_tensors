/*
 * Internal_Shapes.h
 *
 *  Created on: Apr 15, 2018
 *      Author: joseph
 */

#ifndef INTERNAL_SHAPES_H_
#define INTERNAL_SHAPES_H_

#include <vector>
#include <initializer_list>
#include <iostream>
#include "../BlackCat_Internal_Definitions.h"
/*
 * Defines stack array and lambda array
 * These are just two trivially classes used for storing small amounts of homogenous data types, generally ints
 */

namespace BC {

	//This class is more or less a homogenous tuple
	template<class T, int size_>
	struct stack_array : stack_array<T, size_ - 1> {
		static constexpr int SIZE() { return size_; }
		template<class... values>
		__BCinline__ stack_array(T val, values... integers) : stack_array<T, size_ - 1>(integers...), dim(val) {}
		__BCinline__ stack_array() {}

		T dim = 0;
		__BCinline__ auto& next() const { return static_cast<const stack_array<T, size_ - 1>& >(*this); }

		__BCinline__ const T& operator [] (int i) const {
			return (&dim)[-i];
		}
		__BCinline__ T& operator [] (int i) {
			return (&dim)[-i];
		}
		__BCinline__ static constexpr int size() {
			return size_;
		}
	};

	template<class T>
	struct stack_array<T, 0> {
		__BCinline__ T operator [](int i) const {
			std::cout << " out of bounds " << std::endl;
			return 0;
		}
	};

	template<class ref>
	struct lamda_array{
		ref value;

		__BCinline__ lamda_array(ref a) :
				value(a) {
		}

		__BCinline__ const auto operator [](int i) const {
			return value(i);
		}
		__BCinline__ auto operator [](int i) {
			return value(i);
		}
	};

	template<class T> __BChd__
	auto l_array(T data) {
		return lamda_array<T>(data);
	}

	template<class T, class... vals> __BChd__
	auto array(T front, vals... values) {
		return stack_array<T, sizeof...(values) + 1>(front, values...);
	}

	template<class T> struct BlackCat_Shape { static constexpr bool conditional = false; };
	template<> struct BlackCat_Shape<int*> { static constexpr bool conditional = true; };
	template<> struct BlackCat_Shape<const int*> { static constexpr bool conditional = true; };
	template<> struct BlackCat_Shape<const int*&> { static constexpr bool conditional = true; };
	template<class T> struct BlackCat_Shape<lamda_array<T>> { static constexpr bool conditional = true; };
	template<int x> struct BlackCat_Shape<stack_array<int,x>> { static constexpr bool conditional = true; };
	template<> struct BlackCat_Shape<std::vector<int,std::allocator<int>>> { static constexpr bool conditional = true; };

	template<class T> static constexpr bool is_shape = BlackCat_Shape<T>::conditional;

}


#endif /* INTERNAL_SHAPES_H_ */
