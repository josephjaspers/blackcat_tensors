/*
 * BlackCat_Autograd.h
 *
 *  Created on: May 21, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_AUTOGRAD_H_
#define BLACKCAT_AUTOGRAD_H_

#include "BlackCat_Tensors.h"

namespace BC {
namespace autograd {

template<class,class> struct Add;
template<class,class> struct MatMul;

template<class Derived,

template<class,class> class add=Add,
template<class,class> class mul=MatMul
>
struct Expression {

	const Derived& as_derived() const {
		return static_cast<const Derived&>(*this);
	}
	Derived& as_derived()  {
		return static_cast<Derived&>(*this);
	}

	template<class AltDer>
	auto operator + (Expression<AltDer>& expr) {
		 return add<Derived, AltDer>(as_derived(), expr.as_derived());
	}
	template<class AltDer>
	auto operator * (Expression<AltDer>& expr) {
		 return mul<Derived, AltDer>(as_derived(), expr.as_derived());
	}

};

template<class oper, class Lv, class Rv>
struct BinOp  {
	Lv left;
	Rv right;
};


template<class Left, class Right>
struct Add : Expression<Add<Left,Right>> {
	 Left left;
	 Right right;

	 Add(Left l, Right r) : left(l), right(r) {}

	 auto forward() const { return left.forward() + right.forward(); }

	template<class Expression>
	void backward(Expression&& expr) {
		BC::Matrix<float>(expr).print();
		left.backward(expr);
		BC::Matrix<float>(expr).print();

		right.backward(expr);
	}
};
template<class Left, class Right>
struct MatMul : Expression<MatMul<Left, Right>> {
	 Left left;
	 Right right;

	 MatMul(Left l, Right r) : left(l), right(r) {}


	 auto forward() const { return left.forward() * right.forward(); }

	template<class Expression>
	void backward(Expression&& expr) {
		left.backward(expr * right.t());
		right.backward(left.t() * expr);
	}
};

template<class Tensor>
struct Weight : Expression<Weight<Tensor>> {
	Tensor& tensor;
	const Tensor& forward() const { return tensor; }

	Weight(Tensor& te) : tensor(te) {}

	template<class Expression>
	void backward(Expression&& expr) {
		tensor -= expr;
	}
};

template<class Tensor>
struct Result : Expression<Result<Tensor>> {
	Tensor tensor;
	BC::Matrix<float> cache;

	BC::Matrix<float>& forward() {
		return cache=tensor.forward();
	}

	Result(Tensor te) : tensor(te) {
		cache = BC::Matrix<float>(tensor.forward().inner_shape());
	}

	template<class Expression>
	void backward(Expression&& expr) {
		tensor.backward(const_cast<const BC::Matrix<float>&>(cache) - expr);
	}
};

}
}



#endif /* BLACKCAT_AUTOGRAD_H_ */
