/*
 * Common.h
 *
 *  Created on: Jan 13, 2019
 *      Author: joseph
 */

#ifndef BC_STREAMS_COMMON_H_
#define BC_STREAMS_COMMON_H_

#include <memory>

namespace BC {
namespace streams {

template<class derived>
struct QueueInterface {

	QueueInterface() {
		static_assert(!std::is_same<decltype(std::declval<derived>().init()), void>::value,
				"Queue subclass must defined 'void init()'");
		static_assert(!std::is_same<decltype(std::declval<derived>().synchronize()), void>::value,
				"Queue subclass must defined 'void synchronize()'");
		static_assert(!std::is_same<decltype(std::declval<derived>().active()), bool>::value,
				"Queue subclass must defined 'bool active()'");
	}
};


template<class Queue>
class Stream {

	std::shared_ptr<Queue> m_job_queue;

public:

	Stream() : m_job_queue(nullptr) {}
	Stream(const Stream&)=default;
	Stream(Stream&&)=default;

	bool is_default_stream() {
		return bool(m_job_queue);
	}

	void create_stream() {
		m_job_queue = std::shared_ptr<Queue>(new Queue());
		m_job_queue.get()->init();
	}

	void delete_stream() {
		m_job_queue = std::shared_ptr<Queue>(nullptr);
	}

	void sync_stream() {
		//** Pushing a job while syncing is undefined behavior.
		m_job_queue.get()->synchronize();
	}

	template<class function_lambda>
	void push_job(function_lambda functor) {
		m_job_queue.get()->push(functor);
	}

	bool active() {
		return m_job_queue.get() && m_job_queue.get()->active();
	}
};
}
}


#endif /* COMMON_H_ */
