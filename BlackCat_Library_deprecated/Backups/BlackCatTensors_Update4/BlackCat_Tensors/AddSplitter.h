/*
 * AddSplitter.h
 *
 *  Created on: Dec 8, 2017
 *      Author: joseph
 */

#ifndef ADDSPLITTER_H_
#define ADDSPLITTER_H_

constexpr int MAX_THREADS = 8;


template<int value>
constexpr bool isEven() {
	return value % 2;
}

template<int ... dimensions>
struct MATH;

template<int front, int... dimensinos>
struct MATH<front, dimensinos...> {

	template<class T, class U, class V>
	void add(T t, U u, V v) {

	}

};

template<int dim>
struct MATH<dim> {
	template<class T, class U, class V>
	void add(T t, U u, V v) {
		t[dim - 1] = u[dim - 1] + v[dim - 1];
		t[dim - 2] = u[dim - 2] + v[dim - 2];

	}

};

template<int dim>
struct MATH<dim> {
	template<class T, class U, class V>
	void add(T t, U u, V v) {
		t[dim - 1] = u[dim - 1] + v[dim - 1];
		t[dim - 2] = u[dim - 2] + v[dim - 2];

	}

};

#endif /* ADDSPLITTER_H_ */
