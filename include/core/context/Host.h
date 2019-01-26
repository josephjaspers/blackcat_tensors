/*
 * Host.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifndef BC_CONTEXT_HOST_H_
#define BC_CONTEXT_HOST_H_

#include "streams/Streams.h"

namespace BC {
namespace context {

template<class Allocator>
class Host : Allocator {

	using stream_t = streams::Queue<streams::HostQueue>;

	stream_t m_stream;

public:

    const stream_t& get_stream() const {
    	return m_stream;
    }
    stream_t& get_stream() {
    	return m_stream;
    }

    const Allocator& get_allocator() const {
    	return static_cast<const Allocator&>(*this);
    }

    Allocator& get_allocator() {
    	return static_cast<Allocator&>(*this);
    }

    Host() {}
    Host(const Allocator& alloc_) : m_allocator(alloc_) {}
    Host(Allocator&& alloc_) : m_allocator(std::move(alloc_)) {}

    ~Host() {
    }

};


}
}





#endif /* HOST_H_ */
