/*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "structs/forward_list.h"
#include "structs/omp_unique.h"

namespace BC {
namespace NN {

using Structure::forward_list;
using Structure::omp_unique;

using NN_Abreviated_Functions::g; //sigmoid
using NN_Abreviated_Functions::h;
using NN_Abreviated_Functions::gd;
using NN_Abreviated_Functions::hd;


template<class T> using bp_list = omp_unique<forward_list<T>>;

template<class derived>
class Layer {


public:
	scal lr;

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
	struct ClearBPLists {
		template<class T>
		void operator () (T& var) const {
			var.clear();
		}
	};

	static constexpr Sum_Gradients sum_gradients = Sum_Gradients();
	static constexpr Zero_Tensors zero = Zero_Tensors();
	static constexpr ClearBPLists clear	= ClearBPLists();

	auto& xs() { return this->prev().ys(); }




	const int INPUTS;
	const int OUTPUTS = static_cast<derived&>(*this).hasNext() ? this->next().INPUTS : INPUTS;

	Layer(int inputs) : INPUTS(inputs) {
		lr = .03;
	}
//	void set_omp_threads(int i) {
//		next().set_omp_threads(i);
//	}
	auto& next() { return static_cast<derived&>(*this).next(); }
	auto& prev() { return static_cast<derived&>(*this).prev(); }
	const auto& next() const { return static_cast<derived&>(*this).next(); }
	const auto& prev() const { return static_cast<derived&>(*this).prev(); }

	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}
};

}
}



#endif /* LAYER_H_ */
