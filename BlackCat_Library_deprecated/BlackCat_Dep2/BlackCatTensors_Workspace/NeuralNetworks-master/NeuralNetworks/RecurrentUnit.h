#ifndef RecurrentUnitLayer_h
#define RecurrentUnitLayer_h
#include "Layer.h"

class RecurrentUnit : public Layer {

	Vector x;
	Vector b;
	Vector c;
	Matrix w;
	Matrix r;

	Vector b_gradientStorage;
	Matrix w_gradientStorage;
	Matrix r_gradientStorage;

	bpStorage bpX;
	const Vector& Xt();
	bpStorage bpC;
	const Vector& Ct();

public:
	RecurrentUnit(int inputs, int outputs);
	Vector forwardPropagation_express(const Vector& x);
	Vector forwardPropagation(const Vector& x);
	Vector backwardPropagation(const Vector& dy);
	Vector backwardPropagation_ThroughTime(const Vector& dy);

	void clearBPStorage();
	void clearGradients();
	void updateGradients();

	static RecurrentUnit* read(std::ifstream& is);
	void write(std::ofstream& os);
	void writeClass(std::ofstream& os);
};
#endif


