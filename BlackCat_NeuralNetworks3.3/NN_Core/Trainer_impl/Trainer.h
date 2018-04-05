/*
 * Trainer.h
 *
 *  Created on: Apr 5, 2018
 *      Author: joseph
 */

#ifndef TRAINER_H_
#define TRAINER_H_

#include "Trainer_Functions.h"
#include "List.h"

#include <type_traits>

namespace BC {
namespace NN {

class fp;
class bp;
class update;
class clear;


template<class... trainer>
struct Trainer {

};
template<class func, class... func_calls>
struct Trainer<func, func_calls...> : Trainer<func_calls...> {
	template<class T, class U> using IF = std::enable_if_t<std::same<T, U>::value>;
	func f;

	using n = Trainer<func_calls...>;

	const auto& next() const { return static_cast<n&>(*this); }

	template<class NN, class... dl1, typename IF<func, fp>>
	void train(NN& network, data_list<dl1...> list) const {

		auto next_func = [&]() { return next().train(network, list.next()); };
		auto out = network.forwardPropagation(list.data(), next_func);

	}
	template<class NN, class... dl1, typename IF<func, bp>>
	void train(NN& network, data_list<dl1...> list) const {

		auto next_func = [&]() { return next().train(network, list.next()); };
		auto out = network.backPropagation(list.data(), next_func);

	}
};










}
}



#endif /* TRAINER_H_ */
