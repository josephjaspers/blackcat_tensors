#include "stdafx.h"
#include "FeedForward.h"
const Vector & FeedForward::Xt()
{
	if (bpX.empty()) {
		return INPUT_ZEROS;
	}
	return bpX.back();
}

FeedForward::FeedForward(int inputs, int outputs) : Layer(inputs, outputs)
{
	b_gradientStorage = Vector(outputs);
	w_gradientStorage = Matrix(outputs, inputs);

	b = Vector(outputs);
	w = Matrix(outputs, inputs);

	Matrices::randomize(b, -4, 4);
	Matrices::randomize(w, -4, 4);
}

Vector FeedForward::forwardPropagation_express(const Vector & x)
{
	Vector a = w * x + b;
	g(a);

	if (next != nullptr) {
		return next->forwardPropagation_express(a);
	}
	else {
		return a;
	}
}

Vector FeedForward::forwardPropagation(const Vector & x)
{

	bpX.push_back(x); //store the inputs
	Vector a = w * x + b;
	g(a);


	if (next != nullptr)
		return next->forwardPropagation(a);
	else
		return a;
}

Vector FeedForward::backwardPropagation(const Vector & dy)
{
	//Store gradients 
	w_gradientStorage -= (dy * Xt());
	b_gradientStorage -= dy;
	//input delta
	Vector dx = w ->* dy & g.d(Xt());
	//update storage
	bpX.pop_back();
	//continue backprop0
	if (prev != nullptr)
		return prev->backwardPropagation(dx);
	else
		return dx;
}

Vector FeedForward::backwardPropagation_ThroughTime(const Vector & dy)
{	

	//Store gradients 
	w_gradientStorage -= (dy * Xt());
	b_gradientStorage -= dy;
	//input delta
	Vector dx = w ->* dy & g.d(Xt());
	//update storage
	bpX.pop_back();
	//continue backprop
	if (prev != nullptr)
		return prev->backwardPropagation_ThroughTime(dx);
	else
		return dx;
}

void FeedForward::clearBPStorage()
{
	bpX.clear();
	Layer::clearBPStorage();
}

void FeedForward::clearGradients()
{
	Matrix::fill(w_gradientStorage, 0);
	Vector::fill(b_gradientStorage, 0);
	Layer::clearGradients();
}

void FeedForward::updateGradients()
{
	w += w_gradientStorage & lr;
	b += b_gradientStorage & lr;
	Layer::updateGradients();
}

FeedForward* FeedForward::read(std::ifstream & is)
{
	int inputs, outputs;
	is >> inputs;
	is >> outputs;
	FeedForward* ff = new FeedForward(inputs, outputs);

	ff->b = Vector::read(is);
	ff->w = Matrix::read(is);
	ff->g.read(is);

	return ff;
}

void FeedForward::write(std::ofstream & os)
{
	os << NUMB_INPUTS << ' ';
	os << NUMB_OUTPUTS << ' ';

	b.write(os);	//write bias weights
	w.write(os);	//write os weights

	g.write(os);	//write non linearity 
}

void FeedForward::writeClass(std::ofstream & os)
{
	os << 0 << ' ';
}

