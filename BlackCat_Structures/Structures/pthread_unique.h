
#ifndef CONCURRENT_HASH_MAP_H_
#define CONCURRENT_HASH_MAP_H_

#include <pthread.h>
#include "stack_hash_map.h"


//If the pthread_max is not defined we define it ourselves
#ifndef PTHREAD_THREADS_MAX
#define PTHREAD_THREADS_MAX 1024
#endif

namespace BC {
namespace Structure {

template<class T, class deleter = default_deleter>
struct pthread_unique {

	using thread_id_type = decltype(pthread_self());
	using pthread_max_number = decltype(PTHREAD_THREADS_MAX);
	static constexpr pthread_max_number  MAX_THREADS = PTHREAD_THREADS_MAX;


	stack_hash_map<MAX_THREADS, thread_id_type, T> thread_map;

	int size() const {
		return thread_map.size();
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
