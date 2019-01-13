/*
 * Stream.h
 *
 *  Created on: Jan 11, 2019
 *      Author: joseph
 */

#ifndef BC_STREAMS_STREAMS_H_
#define BC_STREAMS_STREAMS_H_

#include "Common.h"
#include "Device.h"
#include "Host.h"


#ifdef __CUDACC__	//------------------------------------------|
																\
namespace BC { 													\
																\
class host_tag;													\
class device_tag;												\
																\
namespace streams {									   	\
																\
	template<class system_tag>									\
	using implementation =										\
			std::conditional_t<									\
				std::is_same<host_tag, system_tag>::value,		\
					Stream<HostQueue>,										\
					Stream<DeviceQueue>>;									\
																\
	}															\
}

#else

																 \
namespace BC { 													 \
																 \
class host_tag;													 \
class device_tag;												 \
																 \
namespace streams {										         \
																 \
	template<													 \
		class system_tag,										 \
		class=std::enable_if<std::is_same<system_tag, host_tag>::value> \
	>															 \
	using implementation = Stream<HostQueue>;				     \
																 \
	}															 \
}
#endif




#endif /* STREAM_H_ */
