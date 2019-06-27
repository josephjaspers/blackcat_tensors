/*
 * Tensor_STL_interface.h
 *
 *  Created on: Oct 18, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSOR_tensor_iterator_dimension_H_
#define BLACKCAT_TENSOR_tensor_iterator_dimension_H_

namespace BC {

template<class internal>
class Tensor_Base;

namespace module {

template<class internal_t>
class Tensor_IterAlgos {

	Tensor_Base<internal_t>& as_derived() {
		return static_cast<Tensor_Base<internal_t>&>(*this);
	}
	const Tensor_Base<internal_t>& as_derived() const {
		return static_cast<const Tensor_Base<internal_t>&>(*this);
	}

public:

	using system_tag = typename internal_t::system_tag;
	using value_type = typename internal_t::value_type;

	//------------------------ iterator-algorthims---------------------------//

    void fill(value_type value) { BC::fill(as_derived().get_stream(), as_derived().begin(), as_derived().end(), value);}
    void zero()                 { fill(0); }
    void ones()                 { fill(1); }

    template<class function>
    void for_each(function func) {
    	as_derived() = as_derived().un_expr(func);
    }
    template<class function>
	void for_each(function func) const {
    	as_derived() = as_derived().un_expr(func);
    }

    void sort() {
    	BC::alg::sort(this->as_derived().get_stream(), this->begin(), this->end());
    }

    void rand(value_type lb=0, value_type ub=1) {
    	randomize(lb, ub);
	}

   void randomize(value_type lb=0, value_type ub=1)  {
	   static_assert(internal_t::tensor_iterator_dimension == 0 || internal_t::tensor_iterator_dimension == 1,
			   	   	   "randomize not available to non-continuous tensors");

	   using Random = BC::random::Random<system_tag>;
	   //Note!! functions and BLAS calls use get_stream, iteralgos use get_stream
	   Random::randomize(this->as_derived().get_stream(), this->as_derived().internal(), lb, ub);
   }

	//------------------------multidimension_iterator------------------------//

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
	auto begin() {
		return iterators::forward_cwise_iterator_begin(as_derived().internal());
	}
	auto end() {
		return iterators::forward_cwise_iterator_end(as_derived().internal());
	}
	const auto cbegin() const {
		return iterators::forward_cwise_iterator_begin(as_derived().internal());
	}
	const auto cend() const {
		return iterators::forward_cwise_iterator_end(as_derived().internal());
	}
	auto rbegin() {
		return iterators::reverse_cwise_iterator_begin(as_derived().internal());
	}
	auto rend() {
		return iterators::reverse_cwise_iterator_end(as_derived().internal());
	}
	const auto crbegin() const {
		return iterators::reverse_cwise_iterator_begin(as_derived().internal());
	}
	const auto crend() const {
		return iterators::reverse_cwise_iterator_end(as_derived().internal());
	}
	//const versions
	auto begin() const {
		return iterators::forward_cwise_iterator_begin(as_derived().internal());
	}
	auto end() const {
		return iterators::forward_cwise_iterator_end(as_derived().internal());
	}
	auto rbegin() const {
		return iterators::reverse_cwise_iterator_begin(as_derived().internal());
	}
	auto rend() const {
		return iterators::reverse_cwise_iterator_end(as_derived().internal());
	}
        //----------------------iterator wrappers---------------------------//

#define BC_TENSOR_tensor_iterator_dimension_DEF(iterator_name, begin_func, end_func)\
    template<class der_t>                                        \
    struct iterator_name {                                       \
                                                                 \
        der_t& tensor;                                           \
                                                                 \
        using begin_t = decltype(tensor.begin_func ());          \
        using end_t = decltype(tensor.end_func ());              \
                                                                 \
        begin_t _begin = tensor.begin_func();                    \
        end_t _end = tensor.end_func();                          \
                                                                 \
        iterator_name(der_t& tensor_) :                          \
                tensor(tensor_) {                                \
        }                                                        \
                                                                 \
        iterator_name(der_t& tensor_, BC::size_t  start) :       \
                tensor(tensor_) {                                \
                                                                 \
            _begin += start;                                     \
        }                                                        \
        iterator_name(der_t& tensor_, BC::size_t  start, BC::size_t  end) :      \
                tensor(tensor_) {                                \
            _begin += start;                                     \
            _end = end;                                          \
        }                                                        \
        auto begin() {                                           \
            return _begin;                                       \
        }                                                        \
        const begin_t& cbegin() const {                          \
            return _begin;                                       \
        }                                                        \
        const end_t& end() const {                               \
            return _end;                                         \
        }                                                        \
                                                                 \
    };                                                           \
                                                                 \
 template<class der_t, class... args>                            \
 static auto make_##iterator_name (der_t& p_derived, args... params) {    \
       return iterator_name<der_t>(p_derived, params...);                 \
 }                                                                        \

BC_TENSOR_tensor_iterator_dimension_DEF(ND_ForwardIterator, nd_begin, nd_end)
BC_TENSOR_tensor_iterator_dimension_DEF(ND_ReverseIterator, nd_rbegin, nd_rend)
BC_TENSOR_tensor_iterator_dimension_DEF(CW_ForwardIterator, begin, end)
BC_TENSOR_tensor_iterator_dimension_DEF(CW_ReverseIterator, rbegin, rend)

#undef BC_TENSOR_tensor_iterator_dimension_DEF

	template<class... params> auto iter(params ... ps) {
		return make_CW_ForwardIterator(as_derived(), ps...);
	}
	template<class... params> auto reverse_iter(params ... ps) {
		return make_CW_ReverseIterator(as_derived(), ps...);
	}
	template<class... params> auto iter(params ... ps) const {
		return make_CW_ForwardIterator(as_derived(), ps...);
	}
	template<class... params> auto reverse_iter(params ... ps) const {
		return make_CW_ReverseIterator(as_derived(), ps...);
	}
	template<class... params> auto nd_iter(params ... ps) {
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
};

}//end of namespace module


#ifdef BC_CPP17 //------------------------------------------------------------------------------------------

template<class internal_t>
using BC_sum_t = std::conditional_t<std::is_same<typename internal_t::value_type, bool>::value,
		BC::size_t, typename internal_t::value_type>;

template<class internal_t>
auto sum(const Tensor_Base<internal_t>& tensor) {
	using sum_value_type = BC_sum_t<internal_t>;
	return BC::accumulate(tensor.cbegin(), tensor.cend(), sum_value_type(0));
}

template<class internal_t>
auto prod(const Tensor_Base<internal_t>& tensor) {
	using value_type = typename internal_t::value_type;
	return BC::accumulate(tensor.cbegin(), tensor.cend(), value_type(1), BC::oper::mul);
}

template<class internal_t>
static bool all(const Tensor_Base<internal_t>& tensor) {
	return tensor.size() == sum(logical(tensor));
}

template<class internal_t>
static bool any(const Tensor_Base<internal_t>& tensor) {
	return sum(logical(tensor)) != 0;
}


template<class internal_t>
static auto max(const Tensor_Base<internal_t>& tensor) {
	static_assert(exprs::expression_traits<internal_t>::is_array,
			"'max' is only available to Array types, max on 'Expressions' is prohibited");
	auto max_index = BC::alg::max_element(tensor.get_stream(), tensor.cbegin(), tensor.cend());
	return tensor(max_index);
}

template<class internal_t>
static auto min(const Tensor_Base<internal_t>& tensor) {
	static_assert(exprs::expression_traits<internal_t>::is_array,
			"'min' is only available to Array types, min on 'Expressions' is prohibited");
	auto min_index = BC::alg::min_element(tensor.get_stream(), tensor.cbegin(), tensor.cend());
	return tensor(min_index);
}

#endif //ifdef BC_CPP17 //------------------------------------------------------------------------------------------

} //end of ns BC

#endif /* TENSOR_STL_INTERFACE_H_ */
