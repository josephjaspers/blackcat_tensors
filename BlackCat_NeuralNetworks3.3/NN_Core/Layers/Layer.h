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
using BC::Structure::thread_map;


template<class T> using bp_list = thread_map<forward_list<T>>;
template<class T> using gradient_list = thread_map<T>;

template<class derived>
class Layer {


public:

	const int INPUTS;
	const int OUTPUTS = next().INPUTS;

	Layer(int inputs) : INPUTS(inputs) {}
	scal lr = scal(.03);

	void init_threads(int i) {
		next().init_threads(i);
	}

	auto& next() {
		return static_cast<derived&>(*this).next();
	}
	auto& prev() {
		return static_cast<derived&>(*this).prev();
	}

	const auto& next() const {
		return static_cast<derived&>(*this).next();
	}
	const auto& prev() const {
		return static_cast<derived&>(*this).prev();
	}

	template<class T> using list = std::list<T>;

};

}



#endif /* LAYER_H_ */
