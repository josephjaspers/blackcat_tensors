/*
 * Get.h
 *
 *  Created on: Jan 22, 2020
 *      Author: joseph
 */

#ifndef BC_TYPE_TRAITS_GET_H_
#define BC_TYPE_TRAITS_GET_H_

#include "constexpr_int.h"
#include "../common.h"
#include <type_traits>
#include <utility>

namespace bc {
namespace traits {
namespace detail {

template<class Arg, class... Args> BCINLINE
auto get_impl(Integer<0>, Arg&& arg, Args&&... args)
	-> decltype(std::forward<Arg>(arg))
{
	return std::forward<Arg>(arg);
}

template<int Index, class Arg, class... Args> BCINLINE
auto get_impl(Integer<Index>, Arg&&, Args&&... args)
	-> decltype(get_impl(Integer<Index-1>(), std::forward<Args>(args)...))
{
	return get_impl(Integer<Index-1>(), std::forward<Args>(args)...);
}

} //ns detail

template<int Index, class... Args> BCINLINE
auto get(Args&&... args)
	-> decltype(detail::get_impl(Integer<Index>(), std::forward<Args>(args)...))
{
	return detail::get_impl(Integer<Index>(), std::forward<Args>(args)...);
}

template<class... Args>
auto get_last(Args&&... args)
	-> decltype(get<sizeof...(Args)-1>(std::forward<Args>(args)...))
{
	return get<sizeof...(Args)-1>(std::forward<Args>(args)...);
}

template<class Arg, class... Args>
auto get_first(Arg&& arg, Args&&... args)
	-> decltype(std::forward<Arg>(arg))
{
	return std::forward<Arg>(arg);
}


}
}



#endif /* GET_H_ */
