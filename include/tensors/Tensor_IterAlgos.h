/*
 * Tensor_STL_interface.h
 *
 *  Created on: Oct 18, 2018
 *	  Author: joseph
 */

#ifndef BLACKCAT_TENSOR_tensor_iterator_dimension_H_
#define BLACKCAT_TENSOR_tensor_iterator_dimension_H_

namespace BC {
namespace tensors {

template<class internal>
class Tensor_Base;

template<class Expression>
class Tensor_IterAlgos {

	Tensor_Base<Expression>& as_derived() {
		return static_cast<Tensor_Base<Expression>&>(*this);
	}

	const Tensor_Base<Expression>& as_derived() const {
		return static_cast<const Tensor_Base<Expression>&>(*this);
	}

public:

	using system_tag = typename Expression::system_tag;
	using value_type = typename Expression::value_type;

	//------------------------ iterator-algorthims---------------------------//

	auto& fill(value_type value) {
		BC::algorithms::fill(
				as_derived().get_stream(), cw_begin(), cw_end(), value);
		return as_derived();
	}

	auto& zero() {
		return fill(0);
	}

	auto& ones() {
		return fill(1);
	}

	template<class function>
	void for_each(function func) {
		as_derived() = as_derived().un_expr(func);
	}

	template<class function>
	void for_each(function func) const {
		as_derived() = as_derived().un_expr(func);
	}

	Tensor_Base<Expression>& sort() {
		BC::algorithms::sort(this->as_derived().get_stream(), this->cw_begin(), this->cw_end());
		return as_derived();
	}

	void rand(value_type lb=0, value_type ub=1) {
		randomize(lb, ub);
	}

   void randomize(value_type lb=0, value_type ub=1)  {
	   static_assert(Expression::tensor_iterator_dimension == 0 || Expression::tensor_iterator_dimension == 1,
			   	   	   "randomize not available to non-continuous tensors");

	   using Random = BC::random::Random<system_tag>;
	   //Note!! functions and BLAS calls use get_stream, iteralgos use get_stream
	   Random::randomize(this->as_derived().get_stream(), this->as_derived().internal(), lb, ub);
   }

	//------------------------multidimension_iterator------------------------//
	auto begin() {
		return iterators::forward_iterator_begin(as_derived());
	}

	auto end() {
		return iterators::forward_iterator_end(as_derived());
	}

	const auto cbegin() const {
		return iterators::forward_iterator_begin(as_derived());
	}

	const auto cend() const {
		return iterators::forward_iterator_end(as_derived());
	}

	auto rbegin() {
		return iterators::reverse_iterator_begin(as_derived());
	}

	auto rend() {
		return iterators::reverse_iterator_end(as_derived());
	}

	const auto crbegin() const {
		return iterators::reverse_iterator_begin(as_derived());
	}

	const auto crend() const {
		return iterators::reverse_iterator_end(as_derived());
	}

	//----------const versions----------//
	auto begin() const {
		return iterators::forward_iterator_begin(as_derived());
	}

	auto end() const {
		return iterators::forward_iterator_end(as_derived());
	}

	auto rbegin() const {
		return iterators::reverse_iterator_begin(as_derived());
	}

	auto rend() const {
		return iterators::reverse_iterator_end(as_derived());
	}

	auto nd_begin() {
		return iterators::forward_iterator_begin(as_derived());
	}

	auto nd_end() {
		return iterators::forward_iterator_end(as_derived());
	}

	const auto nd_cbegin() const {
		return iterators::forward_iterator_begin(as_derived());
	}

	const auto nd_cend() const {
		return iterators::forward_iterator_end(as_derived());
	}

	auto nd_rbegin() {
		return iterators::reverse_iterator_begin(as_derived());
	}

	auto nd_rend() {
		return iterators::reverse_iterator_end(as_derived());
	}

	const auto nd_crbegin() const {
		return iterators::reverse_iterator_begin(as_derived());
	}

	const auto nd_crend() const {
		return iterators::reverse_iterator_end(as_derived());
	}

	//----------const versions----------//
	auto nd_begin() const {
		return iterators::forward_iterator_begin(as_derived());
	}

	auto nd_end() const {
		return iterators::forward_iterator_end(as_derived());
	}

	auto nd_rbegin() const {
		return iterators::reverse_iterator_begin(as_derived());
	}

	auto nd_rend() const {
		return iterators::reverse_iterator_end(as_derived());
	}

	//------------------------elementwise_iterator------------------------//
	auto cw_begin() {
		return iterators::forward_cwise_iterator_begin(as_derived().internal());
	}

	auto cw_end() {
		return iterators::forward_cwise_iterator_end(as_derived().internal());
	}

	const auto cw_cbegin() const {
		return iterators::forward_cwise_iterator_begin(as_derived().internal());
	}

	const auto cw_cend() const {
		return iterators::forward_cwise_iterator_end(as_derived().internal());
	}

	auto cw_rbegin() {
		return iterators::reverse_cwise_iterator_begin(as_derived().internal());
	}

	auto cw_rend() {
		return iterators::reverse_cwise_iterator_end(as_derived().internal());
	}

	const auto cw_crbegin() const {
		return iterators::reverse_cwise_iterator_begin(as_derived().internal());
	}

	const auto cw_crend() const {
		return iterators::reverse_cwise_iterator_end(as_derived().internal());
	}

	//----------const versions----------//
	auto cw_begin() const {
		return iterators::forward_cwise_iterator_begin(as_derived().internal());
	}

	auto cw_end() const {
		return iterators::forward_cwise_iterator_end(as_derived().internal());
	}

	auto cw_rbegin() const {
		return iterators::reverse_cwise_iterator_begin(as_derived().internal());
	}

	auto cw_rend() const {
		return iterators::reverse_cwise_iterator_end(as_derived().internal());
	}
	//----------------------iterator wrappers---------------------------//

#define BC_TENSOR_tensor_iterator_dimension_DEF(iterator_name, begin_func, end_func)\
	template<class der_t>									\
	struct iterator_name {									\
															\
		der_t& tensor;										\
															\
		using begin_t = decltype(tensor.begin_func ());		\
		using end_t = decltype(tensor.end_func ());			\
															\
		begin_t _begin = tensor.begin_func();				\
		end_t _end = tensor.end_func();						\
															\
		iterator_name(der_t& tensor_) :						\
				tensor(tensor_) {							\
		}													\
															\
		iterator_name(der_t& tensor_, BC::size_t  start):	\
				tensor(tensor_) {							\
															\
			_begin += start;								\
		}													\
		iterator_name(der_t& tensor_, BC::size_t  start, BC::size_t  end):	\
				tensor(tensor_) {							\
			_begin += start;								\
			_end = end;										\
		}													\
		auto begin() {										\
			return _begin;									\
		}													\
		const begin_t& cbegin() const {						\
			return _begin;									\
		}													\
		const end_t& end() const {							\
			return _end;									\
		}													\
															\
	};														\
															\
 template<class der_t, class... args>						\
 static auto make_##iterator_name (der_t& p_derived, args... params) {	\
	   return iterator_name<der_t>(p_derived, params...);				\
 }																		\

BC_TENSOR_tensor_iterator_dimension_DEF(ND_ForwardIterator, nd_begin, nd_end)
BC_TENSOR_tensor_iterator_dimension_DEF(ND_ReverseIterator, nd_rbegin, nd_rend)
BC_TENSOR_tensor_iterator_dimension_DEF(CW_ForwardIterator, cw_begin, cw_end)
BC_TENSOR_tensor_iterator_dimension_DEF(CW_ReverseIterator, cw_rbegin, cw_rend)

#undef BC_TENSOR_tensor_iterator_dimension_DEF

	template<class... params> auto cw_iter(params... ps) {
		return make_CW_ForwardIterator(as_derived(), ps...);
	}

	template<class... params> auto cw_reverse_iter(params... ps) {
		return make_CW_ReverseIterator(as_derived(), ps...);
	}

	template<class... params> auto cw_iter(params... ps) const {
		return make_CW_ForwardIterator(as_derived(), ps...);
	}

	template<class... params> auto cw_reverse_iter(params... ps) const {
		return make_CW_ReverseIterator(as_derived(), ps...);
	}

	template<class... params> auto nd_iter(params... ps) {
		return make_ND_ForwardIterator(as_derived(), ps...);
	}

	template<class... params> auto nd_reverse_iter(params ... ps) {
		return make_ND_ReverseIterator(as_derived(), ps...);
	}

	template<class... params> auto nd_iter(params ... ps) const {
		return make_ND_ForwardIterator(as_derived(), ps...);
	}

	template<class... params> auto nd_reverse_iter(params ... ps) const {
		return make_ND_ReverseIterator(as_derived(), ps...);
	}

	template<class... params> auto iter(params ... ps) {
		return nd_iter();
	}

	template<class... params> auto reverse_iter(params ... ps) {
		return nd_reverse_iter();
	}

	template<class... params> auto iter(params ... ps) const {
		return nd_iter();
	}

	template<class... params> auto reverse_iter(params ... ps) const {
		return nd_reverse_iter();
	}

};

#ifdef BC_CPP17

namespace {
template<class Expression>
using BC_sum_t = std::conditional_t<
		std::is_same<typename Expression::value_type, bool>::value,
		BC::size_t,
		typename Expression::value_type>;
}

template<class Expression>
auto value_sum(const Tensor_Base<Expression>& tensor)
{
	using sum_value_type = BC_sum_t<Expression>;
	return BC::algorithms::accumulate(
			BC::streams::select_on_get_stream(tensor),
			tensor.cw_cbegin(),
			tensor.cw_cend(),
			sum_value_type(0));
}

template<class Expression>
auto prod(const Tensor_Base<Expression>& tensor)
{
	using value_type = typename Expression::value_type;
	return BC::algorithms::accumulate(
			BC::streams::select_on_get_stream(tensor),
			tensor.cw_cbegin(),
			tensor.cw_cend(),
			value_type(1),
			BC::oper::mul);
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
	auto max_index = BC::algorithms::max_element(
			BC::streams::select_on_get_stream(tensor),
			tensor.cbegin(),
			tensor.cend());

	return tensor(max_index);
}

template<class Expression>
static auto min(const Tensor_Base<Expression>& tensor)
{
	auto min_index = BC::algorithms::min_element(
			BC::streams::select_on_get_stream(tensor),
			tensor.cbegin(),
			tensor.cend());

	return tensor(min_index);
}

#endif //ifdef BC_CPP17 //------------------------------------------------------------------------------------------
} //end of ns tensors
} //end of ns BC

#endif /* TENSOR_STL_INTERFACE_H_ */
