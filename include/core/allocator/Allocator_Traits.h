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
	using type = std::false_type;
};

//if defined and is 'false_type'
template<class alloc>
struct propagate_on_temporary_construction_of<
	alloc,
	std::enable_if_t<
		!std::is_void<typename alloc::propagate_on_temporary_construction>::value
>>
: std::true_type {

	using type = std::true_type;

};



template<bool, class enabler=void>
struct get_select_on_temporary_construction {
	template<class allocator>
	static auto get(allocator& alloc) {
		return std::allocator_traits<allocator>::select_on_container_copy_construction(alloc);
	}
};
template<>
struct get_select_on_temporary_construction<false, void> {
	template<class allocator>
	static auto get(allocator& alloc) {
		return alloc;
	}
};

//----------------------------allocator_traits-------------------------------------//

template<class Allocator>
struct allocator_traits : std::allocator_traits<Allocator> {
	using system_tag = typename system_tag_of<Allocator>::type;

	using propagate_on_temporary_construction
			= typename propagate_on_temporary_construction_of<Allocator>::type; //true_type or false_type

	static auto select_on_temporary_construction(const Allocator & alloc) {
		return get_select_on_temporary_construction<propagate_on_temporary_construction::value>::get(alloc);
	}

};

}
}



#endif /* ALLOCATOR_TRAITS_H_ */
