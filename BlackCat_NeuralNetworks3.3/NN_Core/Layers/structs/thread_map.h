/*
 * thread_map.h
 *
 *  Created on: Apr 9, 2018
 *      Author: joseph
 */

#ifndef THREAD_MAP_H_
#define THREAD_MAP_H_

#include <unordered_map>
#include <pthread.h>
#include <omp.h>

/*
 * Lightweight structures that creates a set of object and returns based upon the pthread.
 */

namespace BC {
namespace Structure {

struct default_deleter {
	template<class T>
	void operator()(T& t) const {
		return;
	}
};

template<class T, class deleter = default_deleter>
struct thread_map {

	static constexpr deleter destroy = deleter();

	int sz;
	T* pool;

	thread_map(int size = 1) {
		sz = size;
		pool = new T[size];
	}

	int size() const {
		return sz;
	}

	void resize(int size) {
		sz = size;

		for_each(destroy);
		delete[] pool;
		pool = new T[size];
	}

	int valid(int i) {
		if (i < 0 || i > sz - 1) {
			throw std::invalid_argument("pthread_map out of bounds");
		}
		return i;
	}

	int thread_index() {
		return valid(omp_get_thread_num());
	}


	T& operator ()() {
		return pool[thread_index()];
	}
	const T& operator ()() const {
		return pool[thread_index()];
	}

	operator 	   T& ()  	   { return *this(); }
	operator const T& () const { return *this(); }


	template<class functor>
	void for_each(functor function) {
		for (int i = 0; i < size(); ++i) {
			function(pool[i]);
		}
	}

	~thread_map() {
		for_each(destroy);
		delete[] pool;
	}
};

}

}

#endif /* THREAD_MAP_H_ */
