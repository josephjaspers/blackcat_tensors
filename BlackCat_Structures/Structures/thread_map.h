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
		void operator() (T& t) const {
			return;
		}
	};


	template<class T, class deleter = default_deleter>
	struct omp_unique {

		struct set {
			bool initialized = true;
			pthread_t id;
			T data;
		};


		int size() { return sz; }
		int sz;
		deleter destroy;
		set* pool;

		omp_unique(int size) : sz(size) {
			pool = new set[size];
		}

		void resize(int size) {
			sz = size;

			for_each(destroy);
			delete[] pool;
			pool = new set[size];
		}


		void boundsCheck(int i) {
			if (i <0  || i > sz - 1) {
				throw std::invalid_argument("pthread_map out of bounds");
			}
		}


		set* contains(pthread_t thread) const {
			for (int i = 0; i < size(); ++i) {
				if (pool[i].id == thread) {
					&pool[i];
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
			return nullptr;
		}
		void clear() {
			for (set& s : pool) {
				s.initialized = false;
			}
		}


			 T& operator () () 		 {
				 return pool[(int)pthread_self()].data;
		}
		const T& operator () () const {
			return pool[(int)pthread_self()].data;
		}

		template<class functor>
		void for_each(functor function) {
			for (int i = 0; i < size(); ++i) {
				function(pool[i].data);
			}
		}

		~omp_unique() {
			for_each(destroy);
			delete[] pool;
		}
	};


}

}



#endif /* THREAD_MAP_H_ */
