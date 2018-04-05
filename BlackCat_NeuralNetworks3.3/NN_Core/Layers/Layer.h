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

//private:
//	auto& next_() {
//		return static_cast<derived&>(*this).next();
//	}
//	auto& prev_() {
//		return static_cast<derived&>(*this).prev();
//	}
//
//	const auto& next_() const {
//		return static_cast<derived&>(*this).next();
//	}
//	const auto& prev_() const {
//		return static_cast<derived&>(*this).prev();
//	}
//public:
//	template<class T> auto forwardPropagation(const T& x) {
//		return this->next_().forwardPropagation(x);
//	}
//
//	template<class T> auto forwardPropagation_Express(const T& x) const {
//		return this->next_().forwardPropagation_Express(x);
//	}
//
//	template<class T> auto backPropagation(const T& dy) {
//		return this->prev_().backPropagation(dy);
//	}
//
//	auto train(const vec& x, const vec& y) {
//		return this->next_().train(x, y);
//	}
//
//	void updateWeights() {
//		this->next_().updateWeights();
//	}
//	void clearBPStorage() {
//		this->next_().clearBPStorage();
//	}
//
//	void write(std::ofstream& is) const {
//		this->next().write(is);
//	}
//	void read(std::ifstream& os) const {
//		this->next_().read(os);
//	}
//	void setLearningRate(fp_type learning_rate) {
//		this->next_().setLearningRate(learning_rate);
//	}

};

}


#endif /* LAYER_H_ */
