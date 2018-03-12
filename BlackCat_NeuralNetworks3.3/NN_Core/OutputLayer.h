/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef OUTPUTas_CU
#define OUTPUTas_CU

namespace BC {

template<class derived>
struct OutputLayer {

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

public:
	int INPUTS;
	vec hypothesis;

	OutputLayer(int inputs) : INPUTS(inputs), hypothesis(inputs) {
	}

	template<class T> vec forwardPropagation(const vec_expr<T>& in) {
		return hypothesis == in;
	}
	template<class T> vec forwardPropagation_Express(const vec_expr<T>& x) const {
		return x;
	}

	template<class T> vec backPropagation(const vec_expr<T>& y) {
		return prev().backPropagation(hypothesis - y);
	}

	template<class T> auto train(const vec_expr<T>& x, const vec& y) {
		return x - y; //hypothesis - expected
	}

	void updateWeights() {}
	void clearBPStorage() {}
	void write(std::ofstream& is) {
		is << INPUTS << ' ';
		hypothesis.write(is);

	}
	void read(std::ifstream& os) {
		os >> INPUTS;
		hypothesis.read(os);
	}

};
}



#endif /* FEEDFORWARD_CU_ */
