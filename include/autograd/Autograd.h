/*
 * Autograd.h
 *
 *  Created on: Apr 22, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_AUTOGRAD_AUTOGRAD_H_
#define BLACKCAT_AUTOGRAD_AUTOGRAD_H_

namespace BC {
namespace autograd {

template<class Lv, class Rv, class Op>
struct BinaryExpression; //specialize each Op

template<class Tensor, class Op>
struct UnaryExpression; //specialize each Op


template<class Derived>
struct AutoGradExpression {

	template<class Expression>
	auto operator * (AutoGradExpression<Expression>& express) {
		return BinaryExpression<Derived, Expression,
				BC::oper::gemm<typename Expression::system_tag>>(
						this->as_derived(), express.as_derived());
	}
	template<class Expression>
	auto operator + (AutoGradExpression<Expression>& express) {
		return BinaryExpression<
				Derived,
				Expression,
				BC::oper::gemm<typename Expression::system_tag>>(
						this->as_derived(), express.as_derived());
	}

	Derived& as_derived() {
		return static_cast<Derived&>(*this);
	}
};

template<class Lv, class Rv, class Op>
struct BinaryExpression; //specialize each Op

template<class Tensor, class Op>
struct UnaryExpression; //specialize each Op

template<class Tensor>
struct Weight : AutoGradExpression<Weight<Tensor>> {

	using value_type = typename Tensor::value_type;
	using system_tag = typename Tensor::system_tag;

	Tensor tensor;

	template<class... Args>
	Weight(Args... args) : tensor(args...) {}

	value_type learning_rate = 0.003;

	const Tensor& forward() {
		return tensor;
	}

	std::string expression_string() const {
		return "W" + std::to_string(Tensor::DIMS);
	}

	template<class TensorDelta>
	auto backward(TensorDelta tensorDelta) {
		tensor -= tensorDelta;
	}
};

template<class Tensor>
struct Variable : AutoGradExpression<Variable<Tensor>> {
	using value_type = typename Tensor::value_type;
	using system_tag = typename Tensor::system_tag;

	Tensor tensor;
	template<class... Args>
	Variable(Args... args) : tensor(args...) {}

	const Tensor& forward() {
		return tensor;
	}

	std::string expression_string() const {
		return "V" + std::to_string(Tensor::DIMS);
	}

	template<class TensorDelta>
	void backward(TensorDelta tensorDelta) {}

	template<class TensorDelta>
	auto update() {}
};

template<class Lv, class Rv>
struct BinaryExpression<Lv, Rv, BC::oper::add>
: AutoGradExpression<BinaryExpression<Lv, Rv, BC::oper::add>> {

	using system_tag = typename Lv::system_tag;

	Lv& left;
	Rv& right;

	BinaryExpression(Lv& lv_, Rv& rv_) : left(lv_), right(rv_) {}

	const auto forward() {
		return left.forward() + right.forward();
	}

	std::string expression_string() const {
		return left.expression_string() + " + " + right.expression_string();
	}

	template<class TensorDelta>
	auto backward(TensorDelta tensorDelta) {
		left.backward(tensorDelta);
		right.backward(tensorDelta);
	}

	void update() {
		left.update();
		right.update();
	}
};

template<class Lv, class Rv>
struct BinaryExpression<Lv&, Rv&, BC::oper::gemm<typename Lv::system_tag>>
: AutoGradExpression<BinaryExpression<Lv, Rv,BC::oper::gemm<typename Lv::system_tag>>>
{

	using system_tag = typename Lv::system_tag;
	BinaryExpression(Lv& lv_, Rv& rv_) : left(lv_), right(rv_) {}

	Lv& left;
	Rv& right;

	const auto forward() {
		return left.forward() * right.forward();
	}

	std::string expression_string() const {
		return left.expression_string() + " * " + right.expression_string();
	}

	template<class TensorDelta>
	auto backward(TensorDelta tensorDelta) {
		left.backward(tensorDelta * right.t());
		right.backward(left.t() * tensorDelta);
	}

	void update() {
		left.update();
		right.update();
	}
};

template<class Tensor>
struct UnaryExpression<Tensor, BC::module::logistic> :
AutoGradExpression<UnaryExpression<Tensor, BC::module::logistic>>{

	using system_tag = typename Tensor::system_tag;

	Tensor& tensor;
	UnaryExpression(Tensor& tensor_) : tensor(tensor_)
	{}

	const auto forward() {
		return BC::module::logistic(tensor);
	}

	std::string expression_string() const {
		return "logistic(" + tensor.expression_string() + ")";
	}

	template<class TensorDelta>
	auto backward(TensorDelta tensorDelta) {
		return tensor.backward(tensorDelta % BC::module::dx_logistic(tensor));
	}
};


template<class Derived>
auto logistic(AutoGradExpression<Derived> expr) {
	return UnaryExpression<Derived, BC::module::logistic>(expr.as_derived());
}




}
}



#endif /* AUTOGRAD_H_ */
