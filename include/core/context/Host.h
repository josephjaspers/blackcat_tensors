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
struct Host : public Allocator  {

    const Allocator& get_allocator() const {
    	return static_cast<const Allocator&>(*this);
    }

    Allocator& get_allocator() {
    	return static_cast<Allocator&>(*this);
    }

    Host(const Host&) = default;
    Host(Host&&) = default;

    Host(const Allocator& alloc_) : Allocator(alloc_) {}
};


}
}





#endif /* HOST_H_ */
