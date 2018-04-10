#include "stdafx.h"
#include "RecurrentUnit.h"
const Vector & RecurrentUnit::Xt()
{
	if (bpX.empty()) {
		return INPUT_ZEROS;
	}
	return bpX.back();
}

const Vector & RecurrentUnit::Ct()
{
	if (bpC.empty()) {
		return OUTPUT_ZEROS;
	}
	return bpC.back();
}


RecurrentUnit::RecurrentUnit(int inputs, int outputs) : Layer(inputs, outputs)
{
	b_gradientStorage = Vector(outputs);
	w_gradientStorage = Matrix(outputs, inputs);
	r_gradientStorage = Matrix(outputs, outputs);

	x = Vector(inputs);
	c = Vector(outputs);
	b = Vector(outputs);
	w = Matrix(outputs, inputs);
	r = Matrix(outputs, outputs);

	Matrices::randomize(b, -4, 4);
	Matrices::randomize(w, -4, 4);
	Matrices::randomize(r, -4, 0); //Initialize the recurrent weights as having 0 impact initially
}

Vector RecurrentUnit::forwardPropagation_express(const Vector & x)
{
	c = w * x + r * c + b;
	g(c);

	if (next != nullptr) 
		return next->forwardPropagation_express(c);
	else
		return c;
}

Vector RecurrentUnit::forwardPropagation(const Vector & input)
{
	//store the current activations and cellstate
	bpX.push_back(x);
	bpC.push_back(c);
	//set x to inputs
	x = input;
	//set the cellstate (c is the output)
	g(c = (w * x + r * c + b));
	//continue backprop
	if (next != nullptr)
		return next->forwardPropagation(c);
	else
		return c;
}

Vector RecurrentUnit::backwardPropagation(const Vector & dy)
{
	//Store gradients 
	w_gradientStorage -= (dy * x);
	b_gradientStorage -= dy;
	r_gradientStorage -= dy * c; 
	//get input error
	Vector dx = (w ->* dy) & g.d(x);
	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation(dx);
	}
	else
		return dx;
}

Vector RecurrentUnit::backwardPropagation_ThroughTime(const Vector & dy)
{
	//Store gradients 
	w_gradientStorage -= (dy * Xt());
	b_gradientStorage -= dy;
	r_gradientStorage -= dy * Ct();
	//get input error
	Vector dx = (w ->* dy) & g.d(Xt());
	//update backprop storage
	bpX.pop_back();
	bpC.pop_back();
	//continue backpropagation
	if (prev != nullptr) {
		return prev->backwardPropagation_ThroughTime(dx);
	}
	else
		return dx;
}

void RecurrentUnit::clearBPStorage()
{
	bpX.clear();
	bpC.clear();

	Layer::clearBPStorage();
}

void RecurrentUnit::clearGradients()
{
	Matrix::fill(w_gradientStorage, 0);
	Matrix::fill(r_gradientStorage, 0);
	Vector::fill(b_gradientStorage, 0);

	Layer::clearGradients();
}

void RecurrentUnit::updateGradients()
{
	w += w_gradientStorage & lr;
	b += b_gradientStorage & lr;
	r += r_gradientStorage & lr;

	Layer::updateGradients();
}

RecurrentUnit* RecurrentUnit::read(std::ifstream & is)
{
	int inputs, outputs;
	is >> inputs;
	is >> outputs;

	RecurrentUnit* ru = new RecurrentUnit(inputs, outputs);

	ru->x = Vector::read(is);
	ru->b = Vector::read(is);
	ru->c = Vector::read(is);
	ru->w = Matrix::read(is);
	ru->r = Matrix::read(is);

	ru->g.read(is);

	return ru;
}

void RecurrentUnit::write(std::ofstream & os)
{
	os << NUMB_INPUTS << ' ';
	os << NUMB_OUTPUTS << ' ';

	x.write(os);
	b.write(os);
	c.write(os);
	w.write(os);
	r.write(os);

	g.write(os);
}

void RecurrentUnit::writeClass(std::ofstream & os)
{
	os << 3 << ' ';
}
