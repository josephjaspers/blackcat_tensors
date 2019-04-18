/*
 * Context.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifndef BC_CONTEXT_CONTEXT_H_
#define BC_CONTEXT_CONTEXT_H_

#include "Host.h"
#include "Device.h"

BC_DEFAULT_MODULE_BODY(context)

namespace BC {

template<class system_tag>  //push into BC namespace
using Context = context::template implementation<system_tag>;

}



#endif /* CONTEXT_H_ */
