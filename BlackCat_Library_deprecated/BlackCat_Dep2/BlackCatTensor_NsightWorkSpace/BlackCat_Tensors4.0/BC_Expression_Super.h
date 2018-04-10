/*
 * BC_Tensor_Super_Ace.h
 *
 *  Created on: Nov 17, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_ACE_H_
#define BC_TENSOR_SUPER_ACE_H_

class Operation {

	//--------------------------Operations for buffers, they work best with continuous Tensors, though non-continuous functors work as well -------------------//
	template<typename T>
	struct mul {
		template<typename lv, typename rv>
		__attribute__((always_inline)) static T calc(const lv& left, const rv& right) {
			return left * right;
		}
	};
	template<typename T>
	struct div {
		template<typename lv, typename rv>
		__attribute__((always_inline)) static T calc(const lv& left, const rv& right) {
			return left / right;
		}
	};
	template<typename T>
	struct add {
		template<typename lv, typename rv>
		__attribute__((always_inline)) static inline T calc(const lv& left, const rv& right) {
			return left + right;
		}
	};
	template<typename T>
	struct sub {
		template<typename lv, typename rv>
		__attribute__((always_inline)) static T calc(const lv& left, const rv& right) {
			return left - right;
		}
	};
};

template<class T, class lv, class rv, class oper>
class Binary_Expression {

	T* internal_array;
	int* dimensions;




};



#endif /* BC_TENSOR_SUPER_ACE_H_ */
