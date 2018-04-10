/*
 * unq_thread.h
 *
 *  Created on: Aug 24, 2017
 *      Author: joseph
 */

#ifndef UNQ_THREAD_H_
#define UNQ_THREAD_H_
#include <unordered_map>
#include <mutex>
template <typename T, typename ...Types>
class unq_thread {
public:

	mutable std::mutex unq_lock;
	std::unordered_map<pthread_t, T> unq_objects;

	T& operator() () {
		unq_lock.lock();
		T& t_return = unq_objects[pthread_self()];
		unq_lock.unlock();

		return t_return;
	}
	const T& operator() () const {
		unq_lock.lock();
		T& t_return = unq_objects[pthread_self()];
		unq_lock.unlock();

		return t_return;
	}
	T& operator() (pthread_t thread_index) {
		unq_lock.lock();
		T& t_return = unq_objects[thread_index];
		unq_lock.unlock();

		return t_return;
	}
	const T& operator()(pthread_t thread_index) const {
		unq_lock.lock();
		T& t_return = unq_objects[thread_index];
		unq_lock.unlock();

		return t_return;
	}



	void lock() { unq_lock.lock(); }
	void unlock() { unq_lock.unlock(); }
	void clearCache() {
		unq_lock.lock();
		unq_objects.clear();
		unq_lock.unlock();
	}
};


#endif /* UNQ_THREAD_H_ */



















