/*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

namespace BC {
template<class derived>
class Layer {

public:

	const int INPUTS;
	const int OUTPUTS = next().INPUTS;

	Layer(int inputs) : INPUTS(inputs) {}
	scal lr = scal(.03);

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
};

}


#endif /* LAYER_H_ */
