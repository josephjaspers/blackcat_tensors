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

	auto& prev() {
		return static_cast<derived&>(*this).prev();
	}
	const auto& prev() const {
		return static_cast<derived&>(*this).prev();
	}

public:
	int INPUTS;
	int OUTPUTS = INPUTS;
	vec hypothesis;

	OutputLayer(int inputs) : INPUTS(inputs), hypothesis(inputs) {
	}

	template<class T>
	vec forwardPropagation(const _vec<T>& in) {
		return hypothesis == in;
	}
	template<class T>
	vec forwardPropagation_Express(const _vec<T>& x) const {
		return x;
	}

	template<class T>
	vec backPropagation(const _vec<T>& y) {
		return prev().backPropagation(hypothesis - y);
	}

	template<class T>
	auto train(const vec_expr<T>& x, const vec& y) {
		return x - y;
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
	void setLearningRate(fp_type learning_rate) {
		return;
	}

};
}



#endif /* FEEDFORWARD_CU_ */
