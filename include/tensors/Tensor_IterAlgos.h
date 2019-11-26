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

	auto& fill(value_type value) {
		BC::algorithms::fill(
				as_derived().get_stream(), cw_begin(), cw_end(), value);
		return as_derived();
	}

	auto& zero() { return fill(0); }
	auto& ones() { return fill(1); }

	template<class Function>
	void for_each(Function func) const {
		as_derived() = as_derived().un_expr(func);
	}

	template<class Function>
	void for_each(Function func) {
		as_derived() = as_derived().un_expr(func);
	}

	Tensor_Base<Expression>& sort()
	{
		BC::algorithms::sort(
				as_derived().get_stream(), cw_begin(), cw_end());
		return as_derived();
	}

	void randomize(value_type lb=0, value_type ub=1)
	{
		static_assert(
				Expression::tensor_iterator_dimension == 0 ||
				Expression::tensor_iterator_dimension == 1,
				"randomize not available to non-continuous tensors");

		using Random = BC::random::Random<system_tag>;
		Random::randomize(
				this->as_derived().get_stream(),
				this->as_derived().internal(), lb, ub);
	}

#define BC_FORWARD_ITER(suffix, iter, access)              \
	auto suffix##iter() const {                            \
		return iterators::iter_##suffix##iter(access);     \
	}                                                      \
	auto suffix##iter() {                                  \
		return iterators::iter_##suffix##iter(access);     \
	}                                                      \
	auto suffix##c##iter() const {                         \
		return iterators::iter_##suffix##iter(access);     \
	}                                                      \
	auto suffix##r##iter() const {                         \
		return iterators::iter_##suffix##r##iter(access);  \
	}                                                      \
	auto suffix##r##iter() {                               \
		return iterators::iter_##suffix##r##iter(access);  \
	}                                                      \
	auto suffix##cr##iter() const {                        \
		return iterators::iter_##suffix##r##iter(access);  \
	}

	BC_FORWARD_ITER(,begin, as_derived())
	BC_FORWARD_ITER(,end, as_derived())
	BC_FORWARD_ITER(cw_, begin, as_derived().internal())
	BC_FORWARD_ITER(cw_, end, as_derived().internal())

#undef BC_FORWARD_ITER


#define BC_ITERATOR_DEF(suffix, iterator_name, begin_func, end_func)\
	template<class Tensor>											\
	struct iterator_name {											\
																	\
		using size_t = BC::size_t;									\
		Tensor& tensor;												\
																	\
		using begin_t = decltype(tensor.begin_func ());				\
		using end_t = decltype(tensor.end_func ());					\
																	\
		begin_t m_begin = tensor.begin_func();						\
		end_t m_end = tensor.end_func();							\
																	\
		iterator_name(Tensor& tensor) :								\
				tensor(tensor) {}									\
																	\
		iterator_name(Tensor& tensor, size_t start):				\
			tensor(tensor)											\
		{															\
			m_begin += start;										\
		}															\
																	\
		iterator_name(Tensor& tensor, size_t start, size_t end):	\
				tensor(tensor) 										\
		{															\
			m_begin += start;										\
			m_end = end;											\
		}															\
																	\
		auto begin() {												\
			return m_begin;											\
		}															\
																	\
		const begin_t& cbegin() const {								\
			return m_begin;											\
		}															\
																	\
		const end_t& end() const {									\
			return m_end;											\
		}															\
	};																\
																		\
private:																\
																		\
template<class der_t, class... args>									\
static auto make_##iterator_name (der_t& p_derived, args... params) {	\
	return iterator_name<der_t>(p_derived, params...);					\
}																		\
																		\
public:																	\
																		\
template<class... params> auto suffix##iter(params... ps) const {		\
	return make_##iterator_name (as_derived(), ps...);					\
}																		\
																		\
template<class... params> auto suffix##const_iter(params... ps) const {	\
	return make_##iterator_name (as_derived(), ps...);					\
}																		\
																		\
template<class... params> auto suffix##iter(params... ps) {				\
	return make_##iterator_name (as_derived(), ps...);					\
}																		\


BC_ITERATOR_DEF(,nd_iterator_type, begin, end)
BC_ITERATOR_DEF(reverse_, nd_reverse_iterator_type, rbegin, rend)
BC_ITERATOR_DEF(cw_, cw_iterator_type, cw_begin, cw_end)
BC_ITERATOR_DEF(cw_reverse_, cw_reverse_iterator_type, cw_rbegin, cw_rend)

#undef BC_ITERATOR_DEF

};

#ifdef BC_CPP17

template<class Expression>
auto value_sum(const Tensor_Base<Expression>& tensor)
{
	using value_type =  std::conditional_t<
			std::is_same<typename Expression::value_type, bool>::value,
			BC::size_t,
			typename Expression::value_type>;

	return BC::algorithms::accumulate(
			BC::streams::select_on_get_stream(tensor),
			tensor.cw_cbegin(),
			tensor.cw_cend(),
			value_type(0));
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
