/*

 * Common.h
 *
 *  Created on: Jan 13, 2019
 *      Author: joseph
 */

#ifndef BC_STREAMS_COMMON_H_
#define BC_STREAMS_COMMON_H_

#include <memory>
#include "Host_Stream.h"

namespace BC {
namespace streams {

template<class> class Stream;

template<>
class Stream<host_tag> {

	struct Contents {
		std::unique_ptr<HostEvent> m_event;
		HostStream m_stream;
		BC::allocators::Stack_Allocator<host_tag> m_workspace;
	};

	static std::shared_ptr<Contents> get_default_contents() {
		static std::shared_ptr<Contents> default_contents =
				std::shared_ptr<Contents>(new Contents());

		return default_contents;
	}

	std::shared_ptr<Contents> m_contents = get_default_contents();

public:

	using system_tag = host_tag;
	using allocator_type = BC::allocators::Stack_Allocator<host_tag>;

    BC::allocators::Stack_Allocator<host_tag>& get_allocator() {
    	return m_contents->m_workspace;
    }

    template<class RebindType>
    auto get_allocator_rebound() {
    	return typename allocator_type::template rebind<RebindType>::other(m_contents->m_workspace);
    }

    void set_blas_pointer_mode_host() {}
    void set_blas_pointer_mode_device() {}

	bool is_default() {
		return m_contents == get_default_contents();
	}

	void create() {
		m_contents = std::shared_ptr<Contents>(new Contents());
	}

	void destroy() {
		m_contents = get_default_contents();
	}

	void sync() {
		//** Pushing a job while syncing is undefined behavior.
		if (!is_default()) {
			if (!m_contents->m_stream.empty()) {
				record_event();
				this->m_contents->m_event->get_waiter().operator ()();
			}
		}
	}

	void set_stream(Stream stream_) {
		this->m_contents = stream_.m_contents;
	}

    void record_event() {
    	if (!is_default()) {
        	m_contents->m_event = std::unique_ptr<HostEvent>(new HostEvent());
    		this->enqueue(m_contents->m_event->get_recorder());
    	}
    }
    void wait_event(Stream& stream) {
    	if (!stream.is_default()) {
			BC_ASSERT(stream.m_contents->m_event.get(), "Attempting to wait on an event that was never recorded");
			this->enqueue(stream.m_contents->m_event->get_waiter());
    	}
	}

    void wait_stream(Stream& stream) {
    	stream.record_event();
    	this->wait_event(stream);
    }

	template<class Functor>
	void enqueue(Functor functor) {
		if (this->is_default()) {
			host_sync();
			functor();
		} else {
			m_contents->m_stream.push(functor);
		}
	}
	template<class Functor>
	void enqueue_callback(Functor functor) {
		enqueue(functor);
	}

    bool operator == (const Stream& dev) {
    	return m_contents == dev.m_contents;
    }
    bool operator != (const Stream& dev) {
    	return m_contents != dev.m_contents;
    }
};


}
}


#endif /* COMMON_H_ */
