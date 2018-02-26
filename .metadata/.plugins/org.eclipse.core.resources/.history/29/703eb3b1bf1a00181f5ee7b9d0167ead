

/*
 * Shape.h
 *
 *  Created on: Jan 18, 2018
 *      Author: joseph
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include "Tensor_Core_Slice.cu"
#include "Determiners.h"
namespace BC {
template<class T, class Mathlib, class _rank> struct Tensor_Core {


	static constexpr int inner = _rank::inner_rank;
	static constexpr int outer = _rank::outer_rank;

public:
	template<class,class,class> friend class Tensor_Core;

	using self = Tensor_Core<T, Mathlib, _rank>;

	friend class Tensor_Core<T, Mathlib, Rank<higher(inner), outer>>;
	friend class Tensor_Core<T, Mathlib, Rank<lower (inner), outer>>;

	using child = Tensor_Core<T, Mathlib, Rank<lower(inner), outer>>;
	using parent = Tensor_Core<T, Mathlib, Rank<higher(inner), outer>>;

	using dimlist = std::vector<int>;
	using scalar = typename MTF::determine_scalar<T>::type;

	static constexpr int RANK = inner;
	static constexpr int LD_RANK = outer;
	static constexpr bool OWNERSHIP = RANK == LD_RANK;
	static constexpr int LAST = RANK - 1;

	using shape = int[RANK];
	scalar* array;
	shape is;
	shape os;

public:

		  scalar* ary() 	  { return array; }
	const scalar* ary() const { return array; }

	operator 	   scalar*()       { return array; }
	operator const scalar*() const { return array; }

	__BC_gcpu__	      scalar& operator [] (int index) 		{ return array[index]; };
	__BC_gcpu__	const scalar& operator [] (int index) const { return array[index]; };


	Tensor_Core() {
		static_assert(RANK == 0, "DEFAULT CONSTRUCTOR FOR TENSOR_CORE ONLY AVAILABLE FOR RANK == 0 (SCALAR)");
		Mathlib::initialize(array, 1);
	}

	Tensor_Core(dimlist param) {

		if (param.size() != RANK)
			throw std::invalid_argument("dimlist- rank != TENSOR_CORE::RANK");

		if (RANK > 0) {
			Mathlib::copy(is, &param[0], RANK);

			os[0] = is[0];
			for (int i = 1; i < RANK; ++i) {
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, size());
	}

	Tensor_Core(const Tensor_Core& param) {
		if (!OWNERSHIP)	//disable this later
			throw std::invalid_argument("copy on non-ownership tensor");

		Mathlib::copy(is, param.is, RANK);
		Mathlib::copy(os, param.os, RANK);
			os[0] = is[0];
			for (int i = 1; i < RANK; ++i) {
				os[i] = os[i - 1] * is[i];
			}

		Mathlib::initialize(array, size());
		Mathlib::copy(array, param.array, size());
	}
	Tensor_Core(Tensor_Core&& param) {
		if (!OWNERSHIP)	//disable this later
			throw std::invalid_argument("copy on non-ownership tensor");

		Mathlib::copy(is, param.is, RANK);
		Mathlib::copy(os, param.os, RANK);
		array = param.array;
		param.array = nullptr;
	}
	template<class t, class m, class d >
	Tensor_Core(const Tensor_Core<t, m,d >& param) {
		static_assert(RANK == decltype(param)::RANK, "TENSORS MAY ONLY BE CONSTRUCTED OF SAME ORDER ");
		Mathlib::copy(is, param.is, RANK);
		Mathlib::copy(os, param.os, RANK);
		Mathlib::initialize(array, size());
		Mathlib::copy(array, param.array, size());
	}
	~Tensor_Core() {
		if (OWNERSHIP)
			if (array)
		Mathlib::destroy(array);
	}

	__BC_gcpu__ int rank() const { return RANK; }
	__BC_gcpu__ int size() const { return RANK > 0 ? os[LAST] : 1;    }
	__BC_gcpu__ int rows() const { return RANK > 0 ? is[0] : 1; }
	__BC_gcpu__ int cols() const { return RANK > 1 ? is[1] : 1; }
	__BC_gcpu__ int dimension(int i) const { return RANK > i ? is[i] : 1; }

	__BC_gcpu__ int LD_rows() const { return RANK > 0 ? os[0] : 1; }
	__BC_gcpu__ int LD_cols() const { return RANK > 1 ? os[1] : 1; }
	__BC_gcpu__ int LDdimension(int i) const { return RANK > i + 1 ? os[i] : 1; }


	const auto InnerShape() const { return RANK > 0 ? (int*)is : &ONE; }
	const auto OuterShape() const { return RANK > 0 ? (int*)os : &ONE;}

	const scalar* core() const { return array; }
		  scalar* core()  	   { return array; }

	void print() const { Mathlib::print(array, InnerShape(),rank(), 4); }

	void printDimensions() const {
		for (int i = 0; i < RANK; ++i) {
			std::cout << "[" << is[i] << "]";
		}
		std::cout << std::endl;
	}
	void printLDDimensions() const {
		for (int i = 0; i < RANK; ++i) {
			std::cout << "[" << os[i] << "]";
		}
		std::cout << std::endl;
	}

	void resetShape(dimlist sh)  {
		os[0] = sh[0];
		is[0] = sh[0];
		for (int i = 1; i < RANK; ++i) {
			is[i] = sh[i];
			os[i] = os[i - 1] * is[i];
		}
	}

public:
		  auto slice(int i) { return Tensor_Slice<self>(&array[os[LAST - 1] * i], *this); }
	const auto slice(int i) const { return Tensor_Slice<self>(&array[os[LAST - 1] * i], *this); }

};
}

#endif /* SHAPE_H_ */

