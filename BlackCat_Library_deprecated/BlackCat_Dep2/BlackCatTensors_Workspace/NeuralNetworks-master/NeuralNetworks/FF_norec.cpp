#include "stdafx.h"
#include "FF_norec.h"

FF_norec::FF_norec(int inputs, int outputs) : Layer(inputs, outputs)
{
	b_gradientStorage = Vector(outputs);
	w_gradientStorage = Matrix(outputs, inputs);

	b = Vector(outputs);
	w = Matrix(outputs, inputs);
	x = Vector(inputs);
	a = Vector(outputs);

	//Matrices::randomize(b, -4, 4);
	//Matrices::randomize(w, -4, 4);
	b.randomize(-4, 4);
	w.randomize(-4, 4);
}

Vector FF_norec::forwardPropagation_express(const Vector & x)
{
	a = w * x + b;
	g(a);

	if (next != nullptr)
		return next->forwardPropagation_express(a);
	else
		return a;
}

Vector FF_norec::forwardPropagation(const Vector & input)
{
	x = input;
	a = w * x + b;
	g(a);
	if (next != nullptr)
		return next->forwardPropagation(a);
	else
		return a;
}

Vector FF_norec::backwardPropagation(const Vector & dy)
{
	//Store gradients 
	w_gradientStorage -= (dy * x);
	b_gradientStorage -= dy;
	//input delta
	Vector dx = w ->* dy & g.d(x);

	//continue backprop0
	if (prev != nullptr)
		return prev->backwardPropagation(dx);
	else
		return dx;
}

Vector FF_norec::backwardPropagation_ThroughTime(const Vector & dy)
{

	//Store gradients 
	w_gradientStorage -= (dy * x);
	b_gradientStorage -= dy;
	//input delta
	Vector dx = w ->* dy & g.d(x);
	//continue backprop
	if (prev != nullptr)
		return prev->backwardPropagation_ThroughTime(dx);
	else
		return dx;
}

void FF_norec::clearBPStorage()
{
	Layer::clearBPStorage();
}

void FF_norec::clearGradients()
{
	Matrix::fill(w_gradientStorage, 0);
	Vector::fill(b_gradientStorage, 0);
	Layer::clearGradients();
}

void FF_norec::updateGradients()
{
	w += w_gradientStorage & lr;
	b += b_gradientStorage & lr;
	Layer::updateGradients();
}

FF_norec* FF_norec::read(std::ifstream & is)
{
	int inputs, outputs;
	is >> inputs;
	is >> outputs;
	FF_norec* ff = new FF_norec(inputs, outputs);

	ff->b = Vector::read(is);
	ff->w = Matrix::read(is);
	ff->g.read(is);

	return ff;
}

void FF_norec::write(std::ofstream & os)
{
	os << NUMB_INPUTS << ' ';
	os << NUMB_OUTPUTS << ' ';

	b.write(os);	//write bias weights
	w.write(os);	//write os weights

	g.write(os);	//write non linearity 
}

void FF_norec::writeClass(std::ofstream & os)
{
	os << -1 << ' ';
}

