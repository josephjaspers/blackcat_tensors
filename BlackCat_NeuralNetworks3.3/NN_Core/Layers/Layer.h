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
	scal lr = scal(.03);

	struct Sum_Gradients {
		template<class T, class S>
		auto operator () (T& weights, S& lr) const {
			return [&](auto& gradients) { return weights += gradients * lr; };
		}
	};
	struct Zero_Tensors {
		template<class T>
		void operator () (T& var) const {
			var.zero();
		}
	};
	static constexpr Sum_Gradients sum_gradients = Sum_Gradients();
	static constexpr Zero_Tensors zero = Zero_Tensors();





	const int INPUTS;
	const int OUTPUTS = next().INPUTS;

	Layer(int inputs) : INPUTS(inputs) {}

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
