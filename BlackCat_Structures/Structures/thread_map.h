/*
 * thread_map.h
 *
 *  Created on: Apr 9, 2018
 *      Author: joseph
 */

#ifndef THREAD_MAP_H_
#define THREAD_MAP_H_

#include <unordered_map>
#include <pthread.h>
namespace BC {
namespace Structures {


	template<class T, int N = 8>
	struct thread_map {

		T pool[N];

		void boundsCheck(int i) {
			if (i <0  || i > N - 1) {
				throw std::invalid_argument("pthread_map out of bounds");
			}
		}
		T& operator () (pthread_t index) {
			return pool[(int)index];
		}
		const T& operator () (pthread_t index) const {
			return pool[(int)index];
		}
	};


}

}



#endif /* THREAD_MAP_H_ */
