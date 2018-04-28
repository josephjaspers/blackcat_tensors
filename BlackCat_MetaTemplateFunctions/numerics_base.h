/*
 * numerics_base.h
 *
 *  Created on: Apr 27, 2018
 *      Author: joseph
 */

#ifndef NUMERICS_BASE_H_
#define NUMERICS_BASE_H_

namespace BC {
namespace MTF {

//maximum of any number of integers
static constexpr int max(int a) {
	return a;
}

template<class ... ints>
static constexpr int max(int a, ints ... values) {
	return a > max(values...) ? a : max(values...);
}


//minimum of any number of integers
static constexpr int min(int a) {
	return a;
}

template<class ... ints>
static constexpr int min(int a, ints ... values) {
	return a < max(values...) ? a : max(values...);
}









}


}




#endif /* NUMERICS_BASE_H_ */
