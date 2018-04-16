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
 * Lightweight structure that returns a reference to type T with the overloaded function operator().
 * This emulates creating openmp-private variables that will live outside of the scope of parallel section
 *
 * This class only works on openmp spawned threads as it utilizes omp_get_thread_num() for indexing.
 * Past implementations used linear probing and hashing of pthreads, though openmp makes the code much easier
 * possibly will revert to pthread implementation.
 */

namespace BC {
namespace NN {
namespace Structure {

struct default_deleter {
	template<class T>
	void operator()(T& t) const {
		return;
	}
};

template<class T, class deleter = default_deleter>
struct omp_unique {

	static constexpr deleter destroy = deleter();

	int sz;
	T* pool;

	omp_unique(int size = 1) {
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

	~omp_unique() {
		for_each(destroy);
		delete[] pool;
	}
};

}
}
}

#endif /* THREAD_MAP_H_ */
