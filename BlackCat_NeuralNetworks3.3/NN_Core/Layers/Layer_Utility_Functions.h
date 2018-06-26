/*
 * Layer_Utility_Functions.h
 *
 *  Created on: Jun 26, 2018
 *      Author: joseph
 */

#ifndef LAYER_UTILITY_FUNCTIONS_H_
#define LAYER_UTILITY_FUNCTIONS_H_

namespace BC {
namespace NN {


using Structure::forward_list;
using Structure::omp_unique;

using NN_Abreviated_Functions::g; //sigmoid
using NN_Abreviated_Functions::h;
using NN_Abreviated_Functions::gd;
using NN_Abreviated_Functions::hd;
template<class T> using bp_list = omp_unique<forward_list<T>>;


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


}
}



#endif /* LAYER_UTILITY_FUNCTIONS_H_ */
