#include "LinearAlgebraRoutines.h"

template<typename number_type>
void Tensor_Operations<number_type>::power(number_type* store, unsigned s_row, unsigned s_col, unsigned s_inc, number_type* m1, unsigned m1_inc,  number_type* m2, unsigned m2_inc)
{
	unsigned s_base = 0;
	unsigned m1_base = 0;
	unsigned m2_base = 0;

	for (unsigned r = 0; r < s_row; ++r) {
		s_base += s_inc;
		m1_base += m1_inc;
		m2_base += m2_inc;
		for (unsigned c = 0; c < s_col; ++c) {
			store[s_base + c] = pow(m1[m1_base + c], m2[m2_base + c]);
		}
	}
}
template<typename number_type>
void Tensor_Operations<number_type>::multiply(number_type* store, unsigned s_row, unsigned s_col, unsigned s_inc, number_type* m1, unsigned m1_inc,  number_type* m2, unsigned m2_inc)
{
	unsigned s_base = 0;
	unsigned m1_base = 0;
	unsigned m2_base = 0;

	for (unsigned r = 0; r < s_row; ++r) {
		s_base += s_inc;
		m1_base += m1_inc;
		m2_base += m2_inc;
		for (unsigned c = 0; c < s_col; ++c) {
			store[s_base + c] = m1[m1_base + c] * m2[m2_base + c];
		}
	}
}


template<typename number_type>
void Tensor_Operations<number_type>::divide(number_type* store, unsigned s_row, unsigned s_col, unsigned s_inc, number_type* m1, unsigned m1_inc,  number_type* m2, unsigned m2_inc)
{
	unsigned s_base = 0;
	unsigned m1_base = 0;
	unsigned m2_base = 0;

	for (unsigned r = 0; r < s_row; ++r) {
		s_base += s_inc;
		m1_base += m1_inc;
		m2_base += m2_inc;
		for (unsigned c = 0; c < s_col; ++c) {
			store[s_base + c] = m1[m1_base + c] / m2[m2_base + c];
		}
	}
}
template<typename number_type>
void Tensor_Operations<number_type>::add(number_type* store, unsigned s_row, unsigned s_col, unsigned s_inc, number_type* m1, unsigned m1_inc,  number_type* m2, unsigned m2_inc)
{
	unsigned s_base = 0;
	unsigned m1_base = 0;
	unsigned m2_base = 0;

	for (unsigned r = 0; r < s_row; ++r) {
		s_base += s_inc;
		m1_base += m1_inc;
		m2_base += m2_inc;
		for (unsigned c = 0; c < s_col; ++c) {
			store[s_base + c] = m1[m1_base + c] + m2[m2_base + c];
		}
	}
}
template<typename number_type>
void Tensor_Operations<number_type>::subtract(number_type* store, unsigned s_row, unsigned s_col, unsigned s_inc, number_type* m1, unsigned m1_inc,  number_type* m2, unsigned m2_inc)
{
	unsigned s_base = 0;
	unsigned m1_base = 0;
	unsigned m2_base = 0;

	for (unsigned r = 0; r < s_row; ++r) {
		s_base += s_inc;
		m1_base += m1_inc;
		m2_base += m2_inc;
		for (unsigned c = 0; c < s_col; ++c) {
			store[s_base + c] = m1[m1_base + c] - m2[m2_base + c];
		}
	}
}
