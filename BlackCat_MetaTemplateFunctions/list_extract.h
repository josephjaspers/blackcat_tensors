/*
 * list_extractor.h
 *
 *  Created on: Apr 27, 2018
 *      Author: joseph
 */

#ifndef LIST_EXTRACTOR_H_
#define LIST_EXTRACTOR_H_

/*
 * Extracts the types from a collection and injects types into another
 */

namespace BC {
namespace MTF {
template<class...> class list;

//----------------------------------------extract-------------------------------------------/


/*  extract
 *  - Accepts a list of Types and accepts a template type and returns the filled template type
 */

template<class... Ts>
struct extract {
	template<template<class...> class list2> using type = list2<Ts...>;
};


//------------shorthand------------//

template<template< class...> class list2, class T>
 using extract_t = typename extract<T>::type;

}


//----------------------------------------merge-------------------------------------------/


/*
 * merge
 * - Accepts 2 lists and returns the list merged to the second
 * merge_to
 *  - Accepts 2 lists and an empty list and concatenates the elements into the empty list
 */
template<class list1,class list2> struct merge;


template<template<class> class l1, class...Ts,
template<class> class l2, class...Us> struct merge<l1<Ts...>, l2<Us...>> {
	using type 	= l1<Ts..., Us...>;

	template<template<class...> class list>
	using to 	= list<Ts..., Us...>;
};

//------------shorthand------------//

template<class T, class U>
using merge_t = typename merge<T, U>::type;

template<template<class...> class list, class T, class U>
using merge_to_t = typename merge<T,U>::template to<list>;
}




#endif /* LIST_EXTRACTOR_H_ */
