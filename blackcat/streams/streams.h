/*
 * Stream.h
 *
 *  Created on: Jan 24, 2019
 *      Author: joseph
 */

#ifndef BC_CONTEXT_CONTEXT_H_
#define BC_CONTEXT_CONTEXT_H_

#include "../common.h"

BC_DEFAULT_MODULE_BODY(streams, Stream)

#include "stream_synchronization.h"
#include "host.h"
#include "device.h"
#include "logging_stream.h"

namespace bc {

using bc::streams::Stream;

namespace streams {

template<class T>
static auto select_on_get_stream(const T& type) {
	using traits_t = bc::traits::common_traits<T>;
	using system_tag = typename traits_t::system_tag;

	constexpr bool defines_get_stream = traits_t::defines_get_stream::value;

	return bc::traits::constexpr_ternary<defines_get_stream>(
			bc::traits::bind([](const auto& type) {
					return type.get_stream();
			}, type),
			[]() {
					return bc::streams::Stream<system_tag>();
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
