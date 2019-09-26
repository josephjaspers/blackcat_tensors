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

	BC::size_t m_index[N] = { 0 };


	BCINLINE
	value_type size(BC::size_t sz = 1, BC::size_t base_index = N) const {
		return base_index == 0 ?
				sz : size(sz * m_index[base_index - 1], --base_index);
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

	BCINLINE value_type outer_dimension() {
		return this->dimension(N-1);
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

};

template<class... Integers> BCINLINE
auto make_dim(const Integers&... ints) {
	return Dim<sizeof...(Integers)> { ints... };
}


}



#endif /* DIM_H_ */
