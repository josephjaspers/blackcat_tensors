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

	vec forwardPropagation(const vec& in) {
		return hypothesis == in;
	}
	vec forwardPropagation_Express(const vec& x) const {
		return x;
	}
	vec backPropagation(const vec& y) {
		return prev().backPropagation(hypothesis - y);
	}

	void init_threads(int i) {}

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
