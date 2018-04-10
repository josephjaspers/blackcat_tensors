/*
 * BC_Tensor_Super_Ace.h
 *
 *  Created on: Nov 27, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_ACE_H_
#define BC_TENSOR_SUPER_ACE_H_

template<class T, class voider = void>
struct Tensor_Ace {
	using functor_type = T*;

	T* array;

	functor_type data() {
		return array;
	}
	const functor_type data() const {
		return array;
	}
};

template<class T>
struct Tensor_Ace<T, typename std::enable_if<std::is_class<T>::value>::type> {
	using functor_type = T;

	functor_type& data() {
		return static_cast<functor_type&>(*this);
	}
	const functor_type& data() const {
		return static_cast<const functor_type&>(*this);
	}
};

#endif /* BC_TENSOR_SUPER_ACE_H_ */
