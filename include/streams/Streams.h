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
using Stream = streams::Stream<system_tag>;

namespace streams {

template<class T>
static auto select_on_get_stream(const T& type) {
	using system_tag = typename BC::meta::common_traits<T>::system_tag;
	constexpr bool defines_get_stream = BC::meta::common_traits<T>::defines_get_stream;

	static_assert(std::is_same<system_tag, host_tag>::value ||
				std::is_same<system_tag, device_tag>::value, "must be same ");

	return
			meta::constexpr_ternary<defines_get_stream>(
					meta::bind([](const auto& type) {
						return type.get_stream();
			}, type),
			meta::constexpr_else(
					[]() {
						return BC::streams::Stream<system_tag>();
					}
			));
}


}
}



#endif /* CONTEXT_H_ */
