
#ifndef CONCURRENT_HASH_MAP_H_
#define CONCURRENT_HASH_MAP_H_

#include <pthread.h>
#include "stack_hash_map.h"

namespace BC {
namespace Structure {

template<class T, class deleter = default_deleter>
struct pthread_unique {

	using thread_id_type = decltype(pthread_self());
	static constexpr int MAX_THREADS = PTHREAD_THREADS_MAX;


	stack_hash_map<MAX_THREADS, thread_id_type, T> thread_map;

	int size() const {
		return MAX_THREADS; //fixme
	}

	T& operator()() {
		return thread_map[pthread_self()];
	}
	const T& operator()() const {
		return thread_map[pthread_self()];
	}

	operator 	   T& ()  	   { return (*this)(); }
	operator const T& () const { return (*this)(); }

	template<class functor>
	void for_each(functor function) {
		thread_map.for_each(function);
	}
};
}
}



#endif /* CONCURRENT_HASH_MAP_H_ */
