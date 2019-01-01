/*
 * Allocator_Traits.h
 *
 *  Created on: Jan 1, 2019
 *      Author: joseph
 */

#ifndef BC_ALLOCATOR_ALLOCATOR_TRAITS_H_
#define BC_ALLOCATOR_ALLOCATOR_TRAITS_H_

#include <memory>

namespace BC {

class host_tag;
class device_tag;

namespace allocator {


//----------------------system_tag_of---------------------------------//
//If the allocator does not define 'system_tag' default to host_tag
template<class alloc, class enabler=void>
struct system_tag_of : std::false_type { using type = host_tag; };

template<class alloc>
struct system_tag_of<alloc, std::enable_if_t<!std::is_void<typename alloc::system_tag>::value>>
: std::true_type { using type = typename alloc::system_tag; };

//----------------------propagate_on_temporary_construction_of---------------------------------//


//If the allocator does not define 'propagate_on_temporary_construction' OR
//'propagete_on_temporary_construciton' == std::false_type, use the default_allocator

//if not defined
template<class alloc, class enabler=void>
struct propagate_on_temporary_construction_of : std::false_type {
	using system_tag = typename system_tag_of<alloc>::type;
	using value_type = typename alloc::value_type;  //mandatory
	using type = typename allocator::implementation<system_tag, value_type>;
};

//if defined and is 'false_type'
template<class alloc>
struct propagate_on_temporary_construction_of<
	alloc,
	std::enable_if_t<
		!std::is_void<typename alloc::propagate_on_temporary_construction>::value
>>
: std::true_type {

	using prop_type = typename alloc::propagate_on_temporary_construction;
	using system_tag = typename system_tag_of<alloc>::type;
	using value_type = typename alloc::value_type;  //mandatory
	using default_type = typename allocator::implementation<system_tag, value_type>;

	using type = std::conditional_t<std::is_same<std::true_type, prop_type>::value, alloc, //if true use same_type
				 std::conditional_t<std::is_same<std::false_type, prop_type>::value, default_type, //if false use default
				 	 	 	 	 	prop_type>>;	//else use the whatever type is used

};


//----------------------------allocator_traits-------------------------------------//

template<class Allocator>
struct allocator_traits : std::allocator_traits<Allocator> {
	using system_tag = typename system_tag_of<Allocator>::type;

	using  propagate_on_temporary_construction
			= typename propagate_on_temporary_construction_of<Allocator>::type;
};

}
}



#endif /* ALLOCATOR_TRAITS_H_ */
