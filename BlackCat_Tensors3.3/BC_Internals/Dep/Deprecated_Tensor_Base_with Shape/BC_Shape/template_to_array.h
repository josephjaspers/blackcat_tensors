/*
 * template_to_array.h
 *
 *  Created on: Jan 5, 2018
 *      Author: joseph
 */

#ifndef TEMPLATE_TO_ARRAY_H_
#define TEMPLATE_TO_ARRAY_H_

namespace BC {

namespace template_to_array {

	template<class > struct f;

	template<int ... set, template<int...> class list, int front>
	struct f<list<front, set...>> {
		static void fill(int* ary) {
			ary[0] = front;
			f<list<set...>>::fill(&ary[1]);
		}
	};
	template<template<int...> class list, int front>
	struct f<list<front>> {
		static void fill(int* ary) {
			ary[0] = front;
		}
	};
}


}


#endif /* TEMPLATE_TO_ARRAY_H_ */
