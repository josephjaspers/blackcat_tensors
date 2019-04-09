/*

 * Common.h
 *
 *  Created on: Jan 13, 2019
 *      Author: joseph
 */

#ifndef BC_STREAMS_COMMON_H_
#define BC_STREAMS_COMMON_H_

#include <memory>
#include "HostStream.h"

namespace BC {
namespace context {


class Host {

	struct Contents {
		std::unique_ptr<HostEvent> m_event;
		HostStream m_stream;
		BC::allocator::fancy::Scalar_Recycled_Workspace<host_tag> m_workspace;
	};

	static std::shared_ptr<Contents> get_default_contents() {
		thread_local std::shared_ptr<Contents> default_contents =
					std::shared_ptr<Contents>(new Contents());

		return default_contents;
	}
	std::shared_ptr<Contents> m_contents = get_default_contents();

public:

	using system_tag = host_tag;

    BC::allocator::fancy::Scalar_Recycled_Workspace<host_tag>& get_allocator() {
    	return m_contents.get()->m_workspace;
    }

	 Host& get_stream() {
		 return *this;
	 }
	 const Host& get_stream() const {
		 return *this;
	 }

	bool is_default_stream() {
		return m_contents == get_default_contents();
	}

	void create_stream() {
		m_contents = std::shared_ptr<Contents>(new Contents());
		m_contents.get()->m_stream.init();
	}

	void destroy_stream() {
		m_contents = get_default_contents();
	}

	void sync_stream() {
		//** Pushing a job while syncing is undefined behavior.
		if (!is_default_stream()) {
			if (!m_contents.get()->m_stream.empty()) {
				stream_record_event();
				this->m_contents.get()->m_event.get()->get_waiter().operator ()();
			}
		}
	}

	void set_stream(Host& stream_) {
		this->m_contents = stream_.m_contents;
	}

    void stream_record_event() {
    	if (!is_default_stream()) {
        	std::mutex locker;
        	locker.lock();
        	m_contents.get()->m_event = std::unique_ptr<HostEvent>(new HostEvent());
    		this->push_job(m_contents.get()->m_event.get()->get_recorder());
    		locker.unlock();
    	}
    }
    void stream_wait_event(Host& stream) {
    	if (!stream.is_default_stream()) {
			std::mutex locker;
			locker.lock();
			BC_ASSERT(stream.m_contents.get()->m_event.get(), "Attempting to wait on an event that was never recorded");
			this->push_job(stream.m_contents.get()->m_event.get()->get_waiter());
			locker.unlock();
    	}
	}

    void stream_wait_stream(Host& stream) {
    	stream.stream_record_event();
    	this->stream_wait_event(stream);
    }

	template<class Functor>
	void push_job(Functor functor) {
		if (this->is_default_stream()) {
			functor();
		} else {
			m_contents.get()->m_stream.push(functor);
		}
	}
	template<class Functor>
	void stream_enqueue_callback(Functor functor) {
		push_job(functor);
	}

    bool operator == (const Host& dev) {
    	return m_contents == dev.m_contents;
    }
    bool operator != (const Host& dev) {
    	return m_contents != dev.m_contents;
    }
};


}
}


#endif /* COMMON_H_ */
