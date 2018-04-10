#include "stdafx.h"
#include "NeuralNetwork.h"

void NeuralNetwork::push_back(Layer * l)
{
	if (!input) {
		input = l; 
		output = l;
	} else {
		output->link(*l); 
		output = l;
	}
	 ++size;
}

NeuralNetwork::NeuralNetwork() : Layer(0, 0)
{

}

NeuralNetwork::~NeuralNetwork()
{
	if (input != nullptr) {
		delete input;
	}
}

Vector NeuralNetwork::forwardPropagation_express(const Vector & x)
{
	Vector a = input->forwardPropagation_express(x);

	if (next != nullptr) {
		next->forwardPropagation_express(a);
	}
	else {
		return a;
	}
}

Vector NeuralNetwork::forwardPropagation(const Vector & x)
{
	Vector output = input->forwardPropagation(x);
	bpO.push_back(output);

	//if (next != nullptr) {
	//	next->forwardPropagation(output);
	//}
	//else {
		return output;
	//}
}

double sum(Vector& v) {
	double sum = 0;
	for (int i = 0; i < v.length(); ++i) {
		sum += abs(v[i]);
	}
	return sum;
}
Vector NeuralNetwork::backwardPropagation(const Vector & y)
{
	Vector dy = Ot() - y;
	bpO.pop_back();
//	std::cout << " delta sum is " << sum(dy) << std::endl;

	Vector delta = output->backwardPropagation(dy);
	if (prev != nullptr) {
		prev->backwardPropagation(delta);
	}
	else {
		return delta;
	}
}

Vector NeuralNetwork::backwardPropagation_ThroughTime(const Vector & dy)
{
	Vector delta = output->backwardPropagation_ThroughTime(dy);

	if (prev != nullptr) {
		prev->backwardPropagation_ThroughTime(delta);
	}
	else {
		return delta;
	}
}

void NeuralNetwork::train(std::vector<Vector>& x, Vector & y)
{
	clearGradients();
	for (Vector input : x) {
		forwardPropagation(input);
	}
	backwardPropagation(y);
	for (int i = 0; i < x.size() - 1; ++i) {
		backwardPropagation_ThroughTime(output->OUTPUT_ZEROS);
	}

	updateGradients();
	clearBPStorage();
}
void NeuralNetwork::train(std::vector<std::vector<double>> x, std::vector<double> y) {
	clearGradients();
	for (std::vector<double> input : x) {
		bpO.push_back(forwardPropagation(Vector(input)));
	}
	backwardPropagation(Vector(y));
	for (int i = 0; i < x.size() - 1; ++i) {
		backwardPropagation_ThroughTime(output->OUTPUT_ZEROS);
	}
	updateGradients();
	clearBPStorage();
}
void NeuralNetwork::train(std::vector<double> x, std::vector<double> y) {
	clearGradients();
	forwardPropagation(Vector(x));
	backwardPropagation(Vector(y));
	updateGradients();
	clearBPStorage();
}
void NeuralNetwork::train(Vector x, Vector y)
{
	clearGradients();
	forwardPropagation(x);
	backwardPropagation(y);
	updateGradients();
	clearBPStorage();
}
void NeuralNetwork::train(std::vector<Vector>& x, std::vector<Vector>& y)
{
	if (input == nullptr) {
		std::cout << " empty network " << std::endl;
		throw std::invalid_argument("empty network");
	}
	for (int i = 0; i < x.size(); ++i) {
		input->forwardPropagation(x[i]);
	}
	output->backwardPropagation(y.back());
	for (int j = y.size() - 2; j > -1; --j) {
		output->backwardPropagation_ThroughTime(y[j]);
	}
}
void NeuralNetwork::clearBPStorage()
{
	if (input != nullptr)
		input->clearBPStorage();
}
void NeuralNetwork::clearGradients()
{
	if (input != nullptr)
		input->clearGradients();
}
void NeuralNetwork::updateGradients()
{
	if (input != nullptr)
		input->updateGradients();
}
void NeuralNetwork::write(std::ofstream & os)
{
	os << size << ' ';

	if (input != nullptr) {
		input->write_list(os);
	}
}
void NeuralNetwork::writeClass(std::ofstream & os)
{
	os << "NN" << std::endl;
}

NeuralNetwork & NeuralNetwork::read(std::ifstream & is)
{
	if (input) {
		size = 0; //reset 
		delete input;
		input = nullptr;
		output = nullptr;
	}

	std::cout << " reading nn " << std::endl;
	int layers;
	is >> layers;

	for (int i = 0; i < layers; ++i) {
		std::cout << " reading Layer__";
		push_back(readLayer(is));
	}
	return *this;
}

Layer * NeuralNetwork::readLayer(std::ifstream & is)
{
	int classType;
	is >> classType;

	switch (classType) {
	case -1: std::cout << " reading FF_norec" << std::endl; return FF_norec::read(is);
	case 0: std::cout << " reading FF" << std::endl; return FeedForward::read(is);
	case 1: std::cout << " reading GRU" << std::endl; return GRU::read(is);
	case 2: std::cout << " reading LSTM" << std::endl;  return LSTM::read(is);
	case 3: std::cout << " reading RU" << std::endl; return RecurrentUnit::read(is);
	default: std::cout << " class not detected " << classType << std::endl;
	}

	
}
