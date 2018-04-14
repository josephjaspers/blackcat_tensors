/*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_
#include <list>

#include "structs/forward_list.h"
#include "structs/thread_map.h"

namespace BC {
using BC::Structure::forward_list;
using BC::Structure::omp_unique;


template<class T> using bp_list = omp_unique<forward_list<T>>;
template<class T> using gradient_list = omp_unique<T>;

template<class derived>
class BasicLayer {


public:

	BasicLayer(int inputs) : INPUTS(inputs) {}

};

}



#endif /* LAYER_H_ */
