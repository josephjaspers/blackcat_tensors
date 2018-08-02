/*
 * Layer.h
 *
 *  Created on: Aug 1, 2018
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

namespace BC {
namespace NN {

template<template<class...> class prev_l, template<class...> class curr_l, class next_l>
class LayerBase {


	const auto& next() const { return static_cast<const next_l&>(*this).next(); }
		  auto& next() 		 { return static_cast<next_l&>(*this).next(); }
	const auto& prev() const { return static_cast<const derived&>(*this).prev(); }
		  auto& prev() 		 { return static_cast<derived&>(*this).prev(); }




};
}
}

#endif /* LAYER_H_ */
