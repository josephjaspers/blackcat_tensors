#ifndef FF_norec_h
#define FF_norec_h
#include "Layer.h"

class FF_norec : public Layer {

	Vector a;		//activation (output)
	Vector b;		//bias
	Vector x;		//input
	Matrix w;		//weight
	Vector b_gradientStorage;
	Matrix w_gradientStorage;

public:
	FF_norec(int inputs, int outputs);
	Vector forwardPropagation_express(const Vector& x);
	Vector forwardPropagation(const Vector& x);
	Vector backwardPropagation(const Vector& dy);
	Vector backwardPropagation_ThroughTime(const Vector& dy);

	void clearBPStorage();
	void clearGradients();
	void updateGradients();

	static FF_norec* read(std::ifstream& is);
	void write(std::ofstream& os);
	void writeClass(std::ofstream& os);
};
#endif


