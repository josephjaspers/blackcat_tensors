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
};


}
}





#endif /* HOST_H_ */
