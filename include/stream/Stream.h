/*
 * Stream.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifndef BC_CONTEXT_CONTEXT_H_
#define BC_CONTEXT_CONTEXT_H_

BC_DEFAULT_MODULE_BODY(stream, Stream)

#include "Host.h"
#include "Device.h"

namespace BC {

template<class system_tag>  //push into BC namespace
using Stream = stream::Stream<system_tag>;

}



#endif /* CONTEXT_H_ */
