/*
 * Tensor_Static_Functions.h
 *
 *  Created on: Dec 1, 2019
 *      Author: joseph
 */

#ifndef BC_TENSOR_STATIC_FUNCTIONS_H_
#define BC_TENSOR_STATIC_FUNCTIONS_H_

namespace bc {
namespace tensors {

template<class Expression>
auto sum(const Expression_Base<Expression>& tensor) {
	return tensor.un_expr(exprs::Sum<typename Expression::system_tag>());
}

template<class Expression>
auto value_sum(const Expression_Base<Expression>& tensor)
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
auto prod(const Expression_Base<Expression>& tensor)
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
static bool all(const Expression_Base<Expression>& tensor) {
	return tensor.size() == value_sum(logical(tensor));
}

template<class Expression>
static bool any(const Expression_Base<Expression>& tensor) {
	return value_sum(logical(tensor)) != 0;
}

template<class Expression>
static auto max_element(const Expression_Base<Expression>& tensor)
{
	return bc::algorithms::max_element(
			bc::streams::select_on_get_stream(tensor),
			tensor.cw_cbegin(),
			tensor.cw_cend());
}

template<class Expression>
static auto min_element(const Expression_Base<Expression>& tensor)
{
	return bc::algorithms::min_element(
			bc::streams::select_on_get_stream(tensor),
			tensor.cw_cbegin(),
			tensor.cw_cend());
}

template<class Expression>
static bc::size_t max_index(const Expression_Base<Expression>& tensor) {
	return (max_element(tensor) - tensor.data());
}

template<class Expression>
static bc::size_t min_index(const Expression_Base<Expression>& tensor) {
	return min_element(tensor) - tensor.data();
}

template<class Expression>
static auto max(const Expression_Base<Expression>& tensor) {
	return tensor(max_element(tensor));
}

template<class Expression>
static auto min(const Expression_Base<Expression>& tensor) {
	return tensor(min_element(tensor));
}

}
}

#endif /* TENSOR_STATIC_FUNCTIONS_H_ */
