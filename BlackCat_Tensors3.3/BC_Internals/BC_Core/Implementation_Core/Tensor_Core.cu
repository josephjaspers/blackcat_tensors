

/*
 * Shape.h
 *
 *  Created on: Jan 18, 2018
 *      Author: joseph
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include <vector>
#include "../../BC_MetaTemplateFunctions/Adhoc.h"
#include "../../BC_Expressions/BlackCat_Internal_Definitions.h"
#include "Tensor_Core_Specializations/Tensor_Core_Essentials.cu"
#include "Tensor_Core_Specializations/Tensor_Core_Scalar.cu"

#include <iostream>
namespace BC {
template<class T, class Mathlib, class ranker> struct Tensor_Core;

template<class T, class Mathlib, int inner, int outer>
struct Tensor_Core<T, Mathlib, Rank<inner, outer>> {

	using accessor = Tensor_Core<T, Mathlib, Rank<inner - 1, outer>>;

	template<class,class,class> friend class Tensor_Core;

	static constexpr int RANK = inner;
	static constexpr int LD_RANK = outer;
	static constexpr bool tensor_ownership = RANK == LD_RANK;
	static constexpr int BACK = RANK  - 1;

	using _shape = std::vector<int>;
	using scalar = typename MTF::determine_scalar<T>::type;

	scalar* array;
	ifte<tensor_ownership, int[RANK], int*> is;
	ifte<tensor_ownership, int[RANK], int*> os;

		  scalar* ary() { return array; }
	const scalar* ary() const { return array; }

	operator 	   scalar*()       { return array; }
	operator const scalar*() const { return array; }

	__BC_gcpu__	      scalar& operator [] (int index) 		{ return array[index]; };
	__BC_gcpu__	const scalar& operator [] (int index) const { return array[index]; };

	Tensor_Core(_shape param) {
		Mathlib::copy(is, &param[0], RANK);
		if (RANK > 0) {
			os[0] = is[0];
			for (int i = 1; i < RANK; ++i) {
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, size());
	}
	template<class tensor>
	Tensor_Core(const tensor& te) {
		const int* param = te.InnerShape();
				Mathlib::copy(is, &param[0], RANK);
				if (RANK > 0) {
					os[0] = is[0];
					for (int i = 1; i < RANK; ++i) {
						os[i] = os[i - 1] * is[i];
					}
				}
				Mathlib::initialize(array, size());
				Mathlib::copy(array, te.data(), size());
	}

	~Tensor_Core() {
		if (tensor_ownership)
		Mathlib::destroy(array);
	}
//	Tensor_Core(int* param) {
//		Mathlib::copy(is, &param[0], RANK);
//		if (RANK > 0) {
//			os[0] = is[0];
//			for (int i = 1; i < RANK; ++i) {
//				os[i] = os[i - 1] * is[i];
//			}
//		}
//		Mathlib::initialize(array, size());
//	}
	//another constexpr_if replacement (cuda no support, switch to constexpr_if when support available)
	struct ap_scal {
		template<class... params>
		static auto impl(params... p) { return accessor(); }
	};
	struct ap_tensor {
		template<class... params>
		static auto impl(params... p) { return accessor(p...); }
	};

	auto accessor_packet(int index) {
		return std::conditional_t<inner == 1, ap_scal, ap_tensor>::impl(&array[index * os[RANK - 1]], is , os);
	}
	const auto accessor_packet(int index) const {
		return accessor(&array[index * os[RANK - 1]], is , os);
	}
	Tensor_Core(scalar* array, const int* is_, const int * os_)
	: array(array), is(const_cast<int*>(is_)) ,os(const_cast<int*>(os_)) {} //accessor constructor

	Tensor_Core(scalar* ary)
		: array(ary) {}

	__BC_gcpu__ int rank() const { return RANK; }
	__BC_gcpu__ int size() const { return os[BACK];    }
	__BC_gcpu__ int rows() const { return RANK > 0 ? is[0] : 1; }
	__BC_gcpu__ int cols() const { return RANK > 1 ? is[1] : 1; }
	__BC_gcpu__ int dimension(int i) const { return RANK > i ? is[i] : 1; }
	 void printDimensions() const { for (int i = 0; i < RANK; ++i) { std::cout << "["<< is[i] << "]"; } std::cout << std::endl; }
	__BC_gcpu__ void printLDDimensions() const { for (int i = 0; i < RANK; ++i) { std::cout << "["<< os[i] << "]"; } std::cout << std::endl; }
	__BC_gcpu__ int LD_rows() const { return RANK > 0 ? os[0] : 1; }
	__BC_gcpu__ int LD_cols() const { return RANK > 1 ? os[1] : 1; }
	__BC_gcpu__ int LDdimension(int i) const { return RANK > i + 1 ? os[i] : 1; }

	__BC_gcpu__ const int* InnerShape() const { return is; }
	__BC_gcpu__ const int* OuterShape() const { return os; }
	void print() const { Mathlib::print(array, InnerShape(),rank(), 4); }

	__BC_gcpu__ const scalar* data() const { return array; }
	__BC_gcpu__ scalar* data()  		   { return array; }

	void resetShape(_shape sh)  {
		os[0] = sh[0];
		is[0] = sh[0];
		for (int i = 1; i < RANK; ++i) {
			is[i] = sh[i];
			os[i] = os[i - 1] * is[i];
		}
	}
};
}

#endif /* SHAPE_H_ */

