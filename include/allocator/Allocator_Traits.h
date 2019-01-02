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


//----------------------------select_on_temporary_creation-------------------------------------//
//If the allocator does not define 'select_on_temporary_creation' default to host_tag
template<class alloc, class enabler=void>
struct get_select_on_temporary_creation : std::false_type {

	using prop_t = typename propagate_on_temporary_construction_of<alloc>::type;
	static constexpr bool prop_t_is_same = std::is_same<prop_t, alloc>::value;

	//the function to be called if an alternate allocator is not specified
	template<class ADL=void> //ensures two phase lookup
	static std::enable_if_t<std::is_void<ADL>::value && prop_t_is_same, alloc> get(const alloc& alloc_) {
		return std::allocator_traits<alloc>::select_on_copy_construction(alloc_);
	}

	//if a different allocator_type is specified, but select_on is not defined
	template<class ADL=void, class ADL2=void> //ADL2 ensures this counts as a seperate instantiation
	static std::enable_if_t<std::is_void<ADL>::value && prop_t_is_same, alloc> get(const alloc& alloc_) {
		return prop_t();
	}
};


//if select on is defined
template<class alloc>
struct get_select_on_temporary_creation<
	alloc,
	std::enable_if_t<
		!std::is_void<decltype(std::declval<alloc>().select_on_temporary_creation())>::value
	>
> : std::true_type {

	static auto get(const alloc& alloc_) {
		return alloc_.select_on_temporary_creation();
	}

};


//----------------------------allocator_traits-------------------------------------//

template<class Allocator>
struct allocator_traits : std::allocator_traits<Allocator> {
	using system_tag = typename system_tag_of<Allocator>::type;

	using  propagate_on_temporary_construction
			= typename propagate_on_temporary_construction_of<Allocator>::type;

	static auto select_on_temporary_creation(const Allocator& alloc) {
		return get_select_on_temporary_creation<Allocator>::get(alloc);
	}
};

}
}



#endif /* ALLOCATOR_TRAITS_H_ */
