/*
  * Dim.h
 *
 *  Created on: Sep 26, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_SHAPE_DIM_H_
#define BLACKCATTENSORS_SHAPE_DIM_H_

namespace BC {

template<int N>
struct Dim {

	static constexpr int tensor_dimension = N;
	using value_type = BC::size_t;
	using size_t = BC::size_t;

	BC::size_t m_index[N] = { 0 };

	BCINLINE
	value_type size() const {
		return prod();
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
	value_type dimension(BC::size_t i) const {
		//casting to unsigned ensures that negative values do not lead to undefined behavior
		return static_cast<const unsigned&>(i) < N ? m_index[i] : 1;
	}

	BCINLINE value_type outer_dimension() const {
		return this->dimension(N-1);
	}

	BCINLINE bool operator == (const Dim& other) const {
		for (int i = 0; i < N; ++i) {
			if (other[i] != this->m_index[i])
				return false;
		}
		return true;
	}

	BCINLINE bool operator != (const Dim& other) const {
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
	Dim& inplace_op_impl(Operator op, const Dim& other) const {
		for (int i = 0; i < N; ++i) {
			m_index[i] = op(m_index[i], other[i]);
		}
		return *this;
	}

public:
	BCINLINE auto operator + (const Dim& other) const {
		return op_impl(BC::oper::Add(), other);
	}

	BCINLINE auto operator - (const Dim& other) const {
		return op_impl(BC::oper::Sub(), other);
	}

	BCINLINE auto operator / (const Dim& other) const {
		return op_impl(BC::oper::Div(), other);
	}

	BCINLINE auto operator * (const Dim& other) const {
		return op_impl(BC::oper::Mul(), other);
	}

	BCINLINE Dim& operator += (const Dim& other) {
		return inplace_op_impl(BC::oper::add, other);
	}

	BCINLINE Dim& operator -= (const Dim& other) {
		return inplace_op_impl(BC::oper::sub, other);
	}

	BCINLINE Dim& operator /= (const Dim& other) {
		return inplace_op_impl(BC::oper::div, other);
	}

	BCINLINE Dim& operator *= (const Dim& other) {
		return inplace_op_impl(BC::oper::mul, other);
	}

	BCINLINE value_type sum(size_t start=0, size_t end=N) const {
		value_type s = 0;
		for (; start<end; ++start)
			s *= m_index[start];
		return s;
	}

	BCINLINE
	value_type prod(size_t start=0, size_t end=N) const {
		value_type p = 1;
		for (; start<end; ++start)
			p *= m_index[start];
		return p;
	}

	BCINLINE Dim reverse() const {
		Dim rev;
		for (int i = 0; i < N; ++i) {
			rev[i] = m_index[N-1-i];
		}
		return rev;
	}

	template<int Start, int End=N>
	Dim<End-Start> subdim() const {
		return *(reinterpret_cast<const Dim<End-Start>*>(m_index + Start));
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

};

template<class... Integers> BCINLINE
auto dim(const Integers&... ints) {
	return Dim<sizeof...(Integers)> { BC::size_t(ints)... };
}


}



#endif /* DIM_H_ */
