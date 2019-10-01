/*
 * Stream.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifndef BC_CONTEXT_CONTEXT_H_
#define BC_CONTEXT_CONTEXT_H_

BC_DEFAULT_MODULE_BODY(streams, Stream)

#include "Host.h"
#include "Device.h"
#include "Logging_Stream.h"

namespace BC {

using BC::streams::Stream;

namespace streams {

template<class T>
static auto select_on_get_stream(const T& type) {
	using system_tag = typename BC::traits::common_traits<T>::system_tag;
	constexpr bool defines_get_stream =
			BC::traits::common_traits<T>::defines_get_stream::value;

	static_assert(std::is_same<system_tag, host_tag>::value ||
				std::is_same<system_tag, device_tag>::value, "must be same ");

	return traits::constexpr_ternary<defines_get_stream>(
					traits::bind([](const auto& type) {
						return type.get_stream();
			}, type),
			traits::constexpr_else(
					[]() {
						return BC::streams::Stream<system_tag>();
					}
			));
}


}
}



#endif /* CONTEXT_H_ */
