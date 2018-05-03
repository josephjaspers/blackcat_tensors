/*
 * Utility.h
 *
 *  Created on: May 2, 2018
 *      Author: joseph
 */

#ifndef UTILITY_H_
#define UTILITY_H_

namespace BC {

int index_of(int i) {
	return i;
}

template<class... integers>
int index_of(int i, integers... ints) {
	return i * index_of(ints...);
}

}



#endif /* UTILITY_H_ */
