/*
 * BC_Tensor_Primary_Transposer.h
 *
 *  Created on: Nov 26, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_PRIMARY_TRANSPOSER_H_
#define BC_TENSOR_PRIMARY_TRANSPOSER_H_

template<class, class, int, int>
class transpose_expression;

template<class T, class ml, int rows, int cols>
class transpose_expression : public Tensor_Queen<transpose_expression<T, ml, rows, cols>, ml, rows, cols> {
public:
	using this_type = transpose_expression<T, ml, rows, cols>;

	T data;

	transpose_expression<T, ml, rows, cols>(T dat) : data(dat) {}

	__attribute__((always_inline)) auto& operator [](int index) {
		return data[floor(index / rows) + (index % rows) * cols];
	}
	__attribute__((always_inline)) const auto& operator[](int index) const {
		return data[(int) floor(index / rows) + (index % rows) * cols];
	}
};

#endif /* BC_TENSOR_PRIMARY_TRANSPOSER_H_ */
