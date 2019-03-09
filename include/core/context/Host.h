/*

 * Common.h
 *
 *  Created on: Jan 13, 2019
 *      Author: joseph
 */

#ifndef BC_STREAMS_COMMON_H_
#define BC_STREAMS_COMMON_H_

#include <memory>
#include <iostream>
#include "Polymorphic_Allocator.h"
#include "HostQueue.h"


namespace BC {
namespace context {


class Host {

	using Queue = HostQueue;
	std::shared_ptr<Queue> m_job_queue;

public:

	Host() : m_job_queue(nullptr) {}
	Host(const Host&)=default;
	Host(Host&&)=default;

    template<class scalar_t, int value>
    static scalar_t scalar_constant() {
    	return value;
    }

	 template<class scalar_t>
	 scalar_t scalar_alpha(scalar_t val) {
		 return val;
	 }

	 Host& get_stream() {
		 return *this;
	 }
	 const Host& get_stream() const {
		 return *this;
	 }

	bool is_default_stream() {
		return m_job_queue.get() == nullptr;
	}

	void create_stream() {
		m_job_queue = std::shared_ptr<Queue>(new Queue());
		m_job_queue.get()->init();
	}

	void destroy_stream() {
		m_job_queue = std::shared_ptr<Queue>(nullptr);
	}

	void sync_stream() {
		//** Pushing a job while syncing is undefined behavior.
		if (m_job_queue.get())
			m_job_queue.get()->synchronize();
	}

	void set_stream(Host& stream_) {
		this->m_job_queue = stream_.m_job_queue;
	}

	template<class function_lambda>
	void push_job(function_lambda functor) {
		if (this->is_default_stream()) {
			functor();
		} else {
			m_job_queue.get()->push(functor);
		}
	}

    bool operator == (const Host& dev) {
    	return m_job_queue == dev.m_job_queue;
    }
    bool operator != (const Host& dev) {
    	return m_job_queue != dev.m_job_queue;
    }
};


}
}


#endif /* COMMON_H_ */
