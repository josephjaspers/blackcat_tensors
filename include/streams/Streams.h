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
			std::is_same<system_tag, device_tag>::value, "SystemTag Mismatch");

	return traits::constexpr_ternary<defines_get_stream>(
			BC::traits::bind([](const auto& type) {
					return type.get_stream();
			}, type),
			[]() {
					return BC::streams::Stream<system_tag>();
			}
	);
}

template<class>
struct Logging_Stream;

template<class SystemTag>
static auto select_logging_stream(Stream<SystemTag> stream) {
	return Logging_Stream<SystemTag>();
}

template<class SystemTag>
static auto select_logging_stream(Logging_Stream<SystemTag> stream) {
	return stream;
}


}
}


#endif /* CONTEXT_H_ */
