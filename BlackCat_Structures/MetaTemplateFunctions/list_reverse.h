/*
 * MTF_ListFunctions.h
 *
 *  Created on: Apr 27, 2018
 *      Author: joseph
 */

#ifndef MTF_LISTFUNCTIONS_H_
#define MTF_LISTFUNCTIONS_H_

#include "list_extract.h"

namespace BC {
namespace MTF {

/* reverse
 * - Accepts a template type-pack and returns a list of the types in reverse order
 */

template<class...> class list;
//----------------------------------------reverse-------------------------------------------/

template<class...> struct reverse;

template<class T, class... Ts> struct reverse<T, Ts...> {
	using type = merge_t<list<T>, typename reverse<list<Ts...>>::type>;
};

template<class T> struct reverse<T> {
	using type = list<T>;
};

//------------shorthand------------//
template<class... Ts> using reverse_t = typename reverse<Ts...>::type;

}
}



#endif /* MTF_LISTFUNCTIONS_H_ */
