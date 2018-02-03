///*
// * RowVector.h
// *
// *  Created on: Jan 15, 2018
// *      Author: joseph
// */
//
//#ifndef ROWVECTOR_H_
//#define ROWVECTOR_H_
//
//#include "../BC_Shape/Static_Shape.h"
//#include "../BlackCat_Internal_GlobalUnifier.h"
//#include "Scalar.h"
//#include "Matrix.h"
//#include "Tensor_Base.h"
//
//namespace BC {
//
//template<class T, class Mathlib, bool parent>
//class RowVector :
//		public Tensor_Base<T, RowVector<T, Mathlib, parent>,  parent>
////		public Matrix<T, 1, rows, Mathlib, LD>
//{
//	using parent_class = Tensor_Base<T, MATRIX, RowVector<T, rows, Mathlib>, typename default_IS<Mathlib, 1, rows>::type, LD,  Mathlib>;
//	using _int = typename parent_class::subAccess_int;
//	using __int = typename parent_class::force_evaluation_int;
//
//	template<class, int, class, class> friend class Vector;
//
//public:
//	using parent_class::operator=;
//	using parent_class::parent_class;
//
//	auto operator [] (__int i) const {
//		return this->data()[i];
//	}
//
//	Scalar<T, Mathlib> operator[] (_int i) {
//		return (Scalar<T, Mathlib>(&this->array[i]));
//	}
//	const Scalar<T, Mathlib> operator[] (_int i) const {
//		return Scalar<T, Mathlib>(&this->array[i]);
//	}
//
//	template<int R>
//	RowVector<T, R, Mathlib, LD> subVector(int index) {
//		return RowVector<T, R, Mathlib, LD>(&this->array[index]);
//	}
//	template<int R>
//	const RowVector<T, R, Mathlib, LD> subVector(int index) const {
//		return RowVector<T, R, Mathlib, LD>(&this->array[index]);
//	}
//
//	const Vector<T, rows, Mathlib, LD> t() const {
//		return RowVector<T, rows, Mathlib, LD>(this->data());
//	}
//
//	Vector<T, rows, Mathlib, LD> t() {
//		return RowVector<T, rows, Mathlib, LD>(this->data());
//	}
//};
//
//
//
//} //End Namespace BC
//
//
//#endif /* ROWVECTOR_H_ */
