#ifndef FeedForwardLayer_h
#define FeedForwardLayer_h
#include "Layer.h"

class FeedForward : public Layer {

	Vector b;
	Matrix w;
	Vector b_gradientStorage;
	Matrix w_gradientStorage;
	bpStorage bpX;
	const Vector& Xt();

public:
	FeedForward(int inputs, int outputs);
	Vector forwardPropagation_express(const Vector& x);
	Vector forwardPropagation(const Vector& x);
	Vector backwardPropagation(const Vector& dy);
	Vector backwardPropagation_ThroughTime(const Vector& dy);

	void clearBPStorage();
	void clearGradients();
	void updateGradients();

	static FeedForward* read(std::ifstream& is);
	void write(std::ofstream& os);
	void writeClass(std::ofstream& os);
};
#endif


