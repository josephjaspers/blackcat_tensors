/*
 * Host.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifndef BC_CONTEXT_HOST_H_
#define BC_CONTEXT_HOST_H_

#include "stream/Stream.h"

namespace BC {
namespace context {

struct Host : Stream<HostQueue> {

    Host() = default;
    Host(const Host&) = default;
    Host(Host&&) = default;

    template<class scalar_t, int value>
    static scalar_t scalar_constant() {
    	return value;
    }

	 template<class scalar_t>
	 scalar_t scalar_alpha(scalar_t val) {
		 return val;
	 }



	 Stream<HostQueue>& get_stream() {
		 return *this;
	 }
	 const Stream<HostQueue>& get_stream() const {
		 return *this;
	 }

    //These get buffers exist to ensure they match the CUDA interface

	void set_context(Stream<HostQueue>& stream) {
		this->set_stream(stream);
	}
};


}
}





#endif /* HOST_H_ */
