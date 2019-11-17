/*
  * Dim.h
 *
 *  Created on: Sep 26, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_SHAPE_DIM_H_
#define BLACKCATTENSORS_SHAPE_DIM_H_

#include <string>

namespace BC {

template<int N>
struct Dim {

	static_assert(N>=0, "Dim<N>: ASSERT 'N>=0'");

	static constexpr int tensor_dimension = N;
	using value_type = BC::size_t;
	using size_t = BC::size_t;

	BC::size_t m_index[N] = { 0 };

	BCINLINE
	value_type size() const {
		return prod();
	}

	BCINLINE
	Dim& fill(BC::size_t value) {
		for (int i = 0; i < N; ++i)
			m_index[i] = value;
		return *this;
	}

	BCINLINE
	const value_type* data() const {
		return m_index;
	}

	BCINLINE
	value_type* data() {
		return m_index;
	}

	///unchecked version of dimension
	BCINLINE
	const value_type& operator [](int i) const {
		return m_index[i];
	}

	BCINLINE
	value_type& operator [](int i) {
		return m_index[i];
	}

	BCINLINE
	value_type dimension(size_t i, size_t default_value=1) const {
		//casting to unsigned ensures that
		//negative values do not lead to undefined behavior
		return static_cast<const unsigned&>(i) < N ? m_index[i] : default_value;
	}

	BCINLINE
	value_type outer_dimension() const {
		return this->dimension(N-1);
	}

	BCINLINE
	bool operator == (const Dim& other) const {
		for (int i = 0; i < N; ++i) {
			if (other[i] != this->m_index[i])
				return false;
		}
		return true;
	}

	template<
		class... Ints,
		class=std::enable_if_t<
				BC::traits::sequence_of_v<BC::size_t, Ints...>>>
	BCINLINE
	auto concat(Ints... value) const {
		return concat(BC::Dim<sizeof...(Ints)> { value... });
	}

	template <int X> BCINLINE
	Dim<N+X> concat(const Dim<X>& other) const {
		Dim<N+X> concat_dim;

		for (int i = 0; i < N; ++i)
			concat_dim[i] = m_index[i];

		for (int i = 0; i < X; ++i)
			concat_dim[i+N] = other[i];

		return concat_dim;
	}

	BCINLINE
	bool operator != (const Dim& other) const {
		return !(*this == other);
	}

	BCINLINE auto begin() const { return m_index; }
	BCINLINE auto end()   const { return m_index + N; }
	BCINLINE auto begin() { return m_index; }
	BCINLINE auto end()   { return m_index + N; }


private:

	template<class Operator> BCINLINE
	Dim op_impl(Operator op, const Dim& other) const {
		Dim<N> result;
		for (int i = 0; i < N; ++i) {
			result[i] = op(m_index[i], other[i]);
		}
		return result;
	}

	template<class Operator> BCINLINE
	Dim& inplace_op_impl(Operator op, const Dim& other) {
		for (int i = 0; i < N; ++i) {
			m_index[i] = op(m_index[i], other[i]);
		}
		return *this;
	}

	template<class Operator> BCINLINE
	Dim scalar_op_impl(Operator op, const value_type& other) const {
		Dim<N> result;
		for (int i = 0; i < N; ++i) {
			result[i] = op(m_index[i], other);
		}
		return result;
	}

	template<class Operator> BCINLINE
	Dim& inplace_scalar_op_impl(Operator op, const value_type& other) {
		for (int i = 0; i < N; ++i) {
			m_index[i] = op(m_index[i], other);
		}
		return *this;
	}

public:


#define BC_DIM_OP(op, functor)\
	Dim operator op(const Dim& other) const {                           \
		return this->op_impl(BC::oper::functor(), other);                 \
	}

#define BC_DIM_INPLACE_OP(op, functor)\
	Dim operator op##=(const Dim& other) {                              \
		return this->inplace_op_impl(BC::oper::functor(), other);         \
	}

#define BC_DIM_INPLACE_SCALAR_OP(op, functor)                           \
	friend Dim operator op##=(Dim &dim, const value_type& scalar) {     \
		return dim.inplace_scalar_op_impl(BC::oper::functor(), scalar); \
	}                                                                   \

#define BC_DIM_SCALAR_OP(op, functor)                                   \
	friend Dim operator op(const Dim &dim, const value_type& scalar) {  \
		return dim.scalar_op_impl(BC::oper::functor(), scalar);         \
	}                                                                   \
	                                                                    \
	friend Dim operator op(const value_type& scalar, const Dim &dim) {  \
		return dim.scalar_op_impl(BC::oper::functor(), scalar);         \
	}

#define BC_DIM_OP_FACTORY(op, functor) \
BC_DIM_OP(op, functor)                 \
BC_DIM_SCALAR_OP(op, functor)

#define BC_DIM_OP_BOTH(op, functor)   \
BC_DIM_OP_FACTORY(op, functor)        \
BC_DIM_INPLACE_OP(op, functor)        \
BC_DIM_INPLACE_SCALAR_OP(op, functor)

BC_DIM_OP_BOTH(+, Add)
BC_DIM_OP_BOTH(-, Sub)
BC_DIM_OP_BOTH(/, Div)
BC_DIM_OP_BOTH(*, Mul)
BC_DIM_OP_FACTORY(<, Lesser)
BC_DIM_OP_FACTORY(<=, Lesser_Equal)
BC_DIM_OP_FACTORY(>, Greater)
BC_DIM_OP_FACTORY(>=, Greater_Equal)

#undef BC_DIM_OP
#undef BC_DIM_INPLACE_OP
#undef BC_DIM_INPLACE_SCALAR_OP
#undef BC_DIM_SCALAR_OP
#undef BC_DIM_OP_FACTORY
#undef BC_DIM_OP_BOTH

	BCINLINE
	bool all(size_t start, size_t end) const {
		for (; start<end; ++start)
			if (m_index[start] == 0)
				return false;
		return true;
	}

	BCINLINE
	bool all(size_t end=N) const {
		return all(0, end);
	}


	BCINLINE
	value_type sum(size_t start, size_t end) const {
		value_type s = 0;
		for (; start<end; ++start)
			s *= m_index[start];
		return s;
	}

	BCINLINE
	value_type sum(size_t end=N) const {
		return sum(0, end);
	}


	BCINLINE
	value_type prod(size_t start, size_t end) const {
		value_type p = 1;
		for (; start<end; ++start)
			p *= m_index[start];
		return p;
	}

	BCINLINE
	value_type prod(size_t end=N) const {
		return prod(0, end);
	}

	BCINLINE
	Dim reverse() const {
		Dim rev;
		for (int i = 0; i < N; ++i)
			rev[i] = m_index[N-1-i];

		return rev;
	}

	template<int Start, int End=N> BCINLINE
	Dim<End-Start> subdim() const {
		return *(reinterpret_cast<const Dim<End-Start>*>(m_index + Start));
	}

	std::string to_string(int begin, int end) const {
		std::string str= "[";
		while (begin < end) {
			str += std::to_string(m_index[begin++]);

			if (begin != end)
				str += ", ";
		}

		return str + "]";
	}

	std::string to_string(int end=N) const {
		return to_string(0, end);
	}
};



template<>
struct Dim<0> {
	static constexpr int tensor_dimension = 0;
	using value_type = BC::size_t;

	BCINLINE
	value_type size(value_type base_sz=1, value_type base_index=0) const {
		return 1;
	}

	///unchecked version of dimension
	BCINLINE
	value_type operator [](int i) const {
		return 1;
	}

	BCINLINE
	value_type dimension(BC::size_t i) const {
		return 1;
	}

	BCINLINE
	value_type outer_dimension() const {
		return 1;
	}

	BCINLINE bool operator == (const Dim& other) const {
		return true;
	}
	BCINLINE bool operator != (const Dim& other) const {
		return false;
	}

	std::string to_string() const {
		return "[0]";
	}

};

template<class... Integers> BCINLINE
auto dim(const Integers&... ints) {
	return Dim<sizeof...(Integers)> { BC::size_t(ints)... };
}


}



#endif /* DIM_H_ */
