/*
 * Tensor_Static_Functions.h
 *
 *  Created on: Dec 1, 2019
 *      Author: joseph
 */

#ifndef TENSOR_STATIC_FUNCTIONS_H_
#define TENSOR_STATIC_FUNCTIONS_H_

namespace bc {
namespace tensors {

template<class Expression>
auto sum(const Tensor_Base<Expression>& tensor) {
	return tensor.un_expr(exprs::Sum<typename Expression::system_tag>());
}

#ifdef BC_CPP17

template<class Expression>
auto value_sum(const Tensor_Base<Expression>& tensor)
{
	using value_type =  std::conditional_t<
			std::is_same<typename Expression::value_type, bool>::value,
			bc::size_t,
			typename Expression::value_type>;

	return bc::algorithms::accumulate(
			bc::streams::select_on_get_stream(tensor),
			tensor.cw_cbegin(),
			tensor.cw_cend(),
			value_type(0));
}

template<class Expression>
auto prod(const Tensor_Base<Expression>& tensor)
{
	using value_type = typename Expression::value_type;
	return bc::algorithms::accumulate(
			bc::streams::select_on_get_stream(tensor),
			tensor.cw_cbegin(),
			tensor.cw_cend(),
			value_type(1),
			bc::oper::mul);
}

template<class Expression>
static bool all(const Tensor_Base<Expression>& tensor) {
	return tensor.size() == value_sum(logical(tensor));
}

template<class Expression>
static bool any(const Tensor_Base<Expression>& tensor){
	return value_sum(logical(tensor)) != 0;
}


template<class Expression>
static auto max(const Tensor_Base<Expression>& tensor)
{
	auto max_index = bc::algorithms::max_element(
			bc::streams::select_on_get_stream(tensor),
			tensor.cbegin(),
			tensor.cend());

	return tensor(max_index);
}

template<class Expression>
static auto min(const Tensor_Base<Expression>& tensor)
{
	auto min_index = bc::algorithms::min_element(
			bc::streams::select_on_get_stream(tensor),
			tensor.cbegin(),
			tensor.cend());

	return tensor(min_index);
}

#endif

}
}

#endif /* TENSOR_STATIC_FUNCTIONS_H_ */
