/*
 * Tensor_Core_Scalar.h
 *
 *  Created on: Feb 19, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_SCALAR_H_
#define TENSOR_CORE_SCALAR_H_

#include "Tensor_Core_Essentials.cu"
#include <iostream>
#include <vector>

namespace BC {

static const int ONE_SCALAR = 1;

template<class T, class Mathlib, int x>
struct Tensor_Core<T, Mathlib, Rank<0, x>>{

	struct  DISABLED;
	using accessor = DISABLED;

	static constexpr int RANK = 0;
	static constexpr int LD_RANK = x;
	static constexpr bool tensor_ownership = RANK == LD_RANK;
	static constexpr int BACK = RANK  - 1;

	using _shape = std::vector<int>;
	using scalar = typename MTF::determine_scalar<T>::type;

	scalar* array = nullptr;

		  scalar* ary() 	  { return array; }
	const scalar* ary() const { return array; }

	operator 	   scalar*()       { return array; }
	operator const scalar*() const { return array; }

	__BC_gcpu__	      scalar& operator [] (int index) 		{ return array[index]; };
	__BC_gcpu__	const scalar& operator [] (int index) const { return array[index]; };

	Tensor_Core(const Tensor_Core& param) {
		Mathlib::initialize(array, 1);
		Mathlib::copy(array, param.array, 1);
	}
	Tensor_Core() {
		Mathlib::initialize(array, 1);
	}

	Tensor_Core(scalar* ary, int* = nullptr, int* = nullptr) : array(ary)  { }

	__BC_gcpu__ int rank() const { return RANK; }
	__BC_gcpu__ int size() const { return 1;    }
	__BC_gcpu__ int rows() const { return 1; }
	__BC_gcpu__ int cols() const { return 1; }
	__BC_gcpu__ int dimension(int i) const { return 1; }
	 void printDimensions() const { std::cout << "[1]" << std::endl; }
	__BC_gcpu__ void printLDDimensions() const { std::cout << "[0] - Scalar no LD" << std::endl; }
	__BC_gcpu__ int LD_rows() const { return 0; }
	__BC_gcpu__ int LD_cols() const { return 0; }
	__BC_gcpu__ int LDdimension(int i) const { return 0; }

	__BC_gcpu__ const int* InnerShape() const { return &ONE_SCALAR; }
	__BC_gcpu__ const int* OuterShape() const { return &ONE_SCALAR; }
	void print() const { Mathlib::print(array, InnerShape(),rank(), 4); }

	__BC_gcpu__ const scalar* data() const { return array; }
	__BC_gcpu__ scalar* data()  		   { return array; }


private:
	template<class... params>
	Tensor_Core(params...) {
		throw std::invalid_argument("FAILURE -- THIS SHOULD NOT BE CALLED TENSOR_CORE ERROR");
	}
};

template<class T, class Mathlib>
struct Tensor_Core<T, Mathlib, Rank<0, 0>>{

	static constexpr int RANK = 0;
	static constexpr int LD_RANK = 0;
	static constexpr bool tensor_ownership = RANK == LD_RANK;
	static constexpr int BACK = RANK  - 1;

	using _shape = std::vector<int>;
	using scalar = typename MTF::determine_scalar<T>::type;

	struct  DISABLED;
	using accessor = DISABLED;

	scalar* array = nullptr;

		  scalar* ary() 	  { return array; }
	const scalar* ary() const { return array; }

	operator 	   scalar*()       { return array; }
	operator const scalar*() const { return array; }

	__BC_gcpu__	      scalar& operator [] (int index) 		{ return array[index]; };
	__BC_gcpu__	const scalar& operator [] (int index) const { return array[index]; };

	Tensor_Core(const Tensor_Core& param) {
		Mathlib::initialize(array, 1);
		Mathlib::copy(array, param.array, 1);
	}
	Tensor_Core() {
		Mathlib::initialize(array, 1);
	}

	Tensor_Core(scalar* ary, int* = nullptr, int* = nullptr) : array(ary)  { }

	__BC_gcpu__ int rank() const { return RANK; }
	__BC_gcpu__ int size() const { return 1;    }
	__BC_gcpu__ int rows() const { return 1; }
	__BC_gcpu__ int cols() const { return 1; }
	__BC_gcpu__ int dimension(int i) const { return 1; }
	 void printDimensions() const { std::cout << "[1]" << std::endl; }
	 void printLDDimensions() const { std::cout << "[0] - Scalar no LD" << std::endl; }
	__BC_gcpu__ int LD_rows() const { return 0; }
	__BC_gcpu__ int LD_cols() const { return 0; }
	__BC_gcpu__ int LDdimension(int i) const { return 0; }

	__BC_gcpu__ const int* InnerShape() const { return &ONE_SCALAR; }
	__BC_gcpu__ const int* OuterShape() const { return &ONE_SCALAR; }
	void print() const { Mathlib::print(array, InnerShape(),rank(), 4); }

	__BC_gcpu__ const scalar* data() const { return array; }
	__BC_gcpu__ scalar* data()  		   { return array; }


private:
	template<class... params>
	Tensor_Core(params...) {
		throw std::invalid_argument("FAILURE -- THIS SHOULD NOT BE CALLED TENSOR_CORE ERROR");
	}
};
}



#endif /* TENSOR_CORE_SCALAR_H_ */
