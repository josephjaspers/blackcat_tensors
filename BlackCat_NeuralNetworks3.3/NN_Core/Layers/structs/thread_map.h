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
 *
 *
 * Lightweight structures that creates a set of object and returns based upon the pthread.
 *
 *
 *
 *
 *
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

	struct set {
		bool initialized = false;
		pthread_t id;
		T data;
	};


	int sz;
	deleter destroy;
	set* pool;

	thread_map(int size=1) :
			sz(size) {
		pool = new set[size];
	}

	int size() const {
		return sz;
	}

	void resize(int size) {
		sz = size;

		for_each(destroy);
		delete[] pool;
		pool = new set[size];
	}

	void boundsCheck(int i) {
		if (i < 0 || i > sz - 1) {
			throw std::invalid_argument("pthread_map out of bounds");
		}
	}

	set* contains(pthread_t thread) const {
		for (int i = 0; i < size(); ++i) {
			if (pool[i].id == thread) {
				return &pool[i];
			}
		}
		return nullptr;
	}
	set* insert(pthread_t thread) {
		for (int i = 0; i < size(); ++i) {
			if (!pool[i].initialized) {
				pool[i].id = thread;
				pool[i].initialized = true;
				return &pool[i];
			}
		}
		std::cout << " insertion failure insert(pthread_t) thread_map " << std::endl;
		return nullptr;
	}
	void clear() {
		for (int i = 0; i < size(); ++i) {
			pool[i].initialized = false;
		}
	}

	T& operator ()() {
		return pool[omp_get_thread_num()].data;
		std::cout << " returning null - operator() thread_map" << std::endl;
		std::cout << " number of threads exceeds current size" << std::endl;

	}

	template<class functor>
	void for_each(functor function) {
		for (int i = 0; i < size(); ++i) {
			function(pool[i].data);
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
