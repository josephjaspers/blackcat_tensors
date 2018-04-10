///*
// * Scalar.h
// *
// *  Created on: Oct 11, 2017
// *      Author: joseph
// */
//
//#ifndef SCALAR_H_
//#define SCALAR_H_
//
//template <class M, class b>
//class buffer;
//
//template<typename T, class Math>
//class Scalar : public Tensor_Super<T, Math> {
//
//	T* data;
//
//	Scalar<T, Math> operator &(const Scalar<T, Math>& scal) const;
//	Scalar<T, Math> operator /(const Scalar<T, Math>& scal) const;
//	Scalar<T, Math> operator +(const Scalar<T, Math>& scal) const;
//	Scalar<T, Math> operator -(const Scalar<T, Math>& scal) const;
//
//	Scalar<T, Math>& operator &=(const Scalar<T, Math>& scal) const;
//	Scalar<T, Math>& operator /=(const Scalar<T, Math>& scal) const;
//	Scalar<T, Math>& operator +=(const Scalar<T, Math>& scal) const;
//	Scalar<T, Math>& operator -=(const Scalar<T, Math>& scal) const;
//
//};
//
//template<typename T, class Oper>
//class Vector : public Scalar<T, Oper> {
//
//	Buffer<T, Oper> operator *(const Vector<T, Oper>& vec) const;
//	Buffer<T, Oper> operator ->* (const Vector<T,Oper>& vec) const;
//
//	Buffer<T, Oper> operator &(const Scalar<T, Oper>& scal) const;
//	Buffer<T, Oper> operator /(const Scalar<T, Oper>& scal) const;
//	Buffer<T, Oper> operator +(const Scalar<T, Oper>& scal) const;
//	Buffer<T, Oper> operator -(const Scalar<T, Oper>& scal) const;
//
//	Vector<T, Oper>& operator &=(const Scalar<T, Oper>& scal) const;
//	Vector<T, Oper>& operator /=(const Scalar<T, Oper>& scal) const;
//	Vector<T, Oper>& operator +=(const Scalar<T, Oper>& scal) const;
//	Vector<T, Oper>& operator -=(const Scalar<T, Oper>& scal) const;
//
//	Buffer<T, Oper> operator &(const Vector<T, Oper>& vec) const;
//	Buffer<T, Oper> operator /(const Vector<T, Oper>& vec) const;
//	Buffer<T, Oper> operator +(const Vector<T, Oper>& vec) const;
//	Buffer<T, Oper> operator -(const Vector<T, Oper>& vec) const;
//
//	Vector<T, Oper>& operator &=(const Vector<T, Oper>& vec) const;
//	Vector<T, Oper>& operator /=(const Vector<T, Oper>& vec) const;
//	Vector<T, Oper>& operator +=(const Vector<T, Oper>& vec) const;
//	Vector<T, Oper>& operator -=(const Vector<T, Oper>& vec) const;
//
//};
//
//template<typename T, class Oper>
//class Matrix : public Scalar<T, Oper> {
//
//	Buffer<T, Oper> operator &(const Scalar<T, Oper>& scal) const;
//	Buffer<T, Oper> operator /(const Scalar<T, Oper>& scal) const;
//	Buffer<T, Oper> operator +(const Scalar<T, Oper>& scal) const;
//	Buffer<T, Oper> operator -(const Scalar<T, Oper>& scal) const;
//
//	Matrix<T, Oper>& operator &=(const Scalar<T, Oper>& scal) const;
//	Matrix<T, Oper>& operator /=(const Scalar<T, Oper>& scal) const;
//	Matrix<T, Oper>& operator +=(const Scalar<T, Oper>& scal) const;
//	Matrix<T, Oper>& operator -=(const Scalar<T, Oper>& scal) const;
//
//	Buffer<T, Oper> operator &(const Vector<T, Oper>& scal) const;
//	Buffer<T, Oper> operator /(const Vector<T, Oper>& scal) const;
//	Buffer<T, Oper> operator +(const Vector<T, Oper>& scal) const;
//	Buffer<T, Oper> operator -(const Vector<T, Oper>& scal) const;
//
//	Vector<T, Oper>& operator &=(const Vector<T, Oper>& scal) const;
//	Vector<T, Oper>& operator /=(const Vector<T, Oper>& scal) const;
//	Vector<T, Oper>& operator +=(const Vector<T, Oper>& scal) const;
//	Vector<T, Oper>& operator -=(const Vector<T, Oper>& scal) const;
//
//};
//
//
//#endif /* SCALAR_H_ */
//class CPU {
//
//	template<typename T, typename lv, typename rv, struct math>
//	struct Pointwise_Operation_Buffer {
//
//		rv* curr;
//		lv* next;
//
//		unsigned sz;
//
//		void eval(T* assign_to) {
//			for (int index = 0; index < sz; ++index)
//			assign_to[index] = math(index);
//		}
//		virtual T math(unsigned index) = 0;
//	};
//
//	template<typename T, typename lv, typename rv>
//	struct mul : Operation<T, lv, rv> {
//		T math(unsigned index) override final {
//			return curr[index] * next[index];
//		}
//	};
//	template<typename T>
//	struct div : Pointwise_Operation_Buffer<T> {
//		T math(unsigned index) override final {
//			return curr[index] / next[index];
//		}
//	};
//	template<typename T>
//	struct add : Pointwise_Operation_Buffer<T> {
//		T math(unsigned index) override final {
//			return curr[index] + next[index];
//		}
//	};
//	template<typename T>
//	struct sub : Pointwise_Operation_Buffer<T> {
//		T math(unsigned index) override final {
//			return curr[index] - next[index];
//		}
//	};
//
//	//-----------------------for the scalar operations ------------- "next" is always the scalar
//
//	template<typename T>
//	struct mul_scal : Pointwise_Operation_Buffer<T> {
//		T math(unsigned index) override final {
//			return curr[index] * (*next);
//		}
//	};
//	template<typename T>
//	struct div_scal : Pointwise_Operation_Buffer<T> {
//		T math(unsigned index) override final {
//			return curr[index] / (*next);
//		}
//	};
//	template<typename T>
//	struct add_scal : Pointwise_Operation_Buffer<T> {
//		T math(unsigned index) override final {
//			return curr[index] + (*next);
//		}
//
//	};
//	template<typename T>
//	class sub_scal : Pointwise_Operation_Buffer<T> {
//		T math(unsigned index) override final {
//			return curr[index] - (*next);
//		}
//	};
//};
