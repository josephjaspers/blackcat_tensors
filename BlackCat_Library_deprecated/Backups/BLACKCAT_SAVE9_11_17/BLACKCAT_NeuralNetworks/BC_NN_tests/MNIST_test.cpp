
#include "Tensor.h"
#include "FeedForward.h"
#include "NeuralNetwork.h"
#include "ConvolutionalLayer.h"
#include <fstream>
#include <thread>
#include <atomic>
#include <list>
typedef std::vector<Vector<double>> data;

Tensor<double> expandOutput(int val, int total) {
	Tensor<double> out(total);
	out.fill(0);
	out(val) = 1;
	return out;
}

Tensor<double> normalize(Tensor<double> tens, double max, double min) {
	for (unsigned i = 0; i < tens.size(); ++i) {
		tens(i) = (tens(i) - min) / (max - min);
		if (tens(i) > 0 && tens(i) < .0001) {
			tens(i) = 0;
		}
	}
	return tens;
}

void generateAndLoad(data& input_data, data& output_data, std::ifstream& read_data, unsigned MAXVALS) {
	unsigned numb = 0;
	unsigned vals = 0;
	while (read_data.good() && vals < MAXVALS) {
		Tensor<double> input;
		input.readCSV(read_data, 785);

		input_data.push_back(normalize(input({1}, {784}), 255, 0));
		output_data.push_back(expandOutput(input(0), 10));

		++vals;
	}
}

void doWork(Tensor<double>& a) {
	a({0, 2, 2},{5, 5, 2}) += 1;

	(a({0, 2, 2},{2, 2, 2}) += 1) += Tensor<double>(2,2,1);

	a &= 3;

}

int main() {
std::cout << " main";
	//create network
	std::cout << " generating network" << std::endl;
//	ConvolutionalLayer f1(28, 28,1, 5, 5, 5);	//create FF layers
//	ConvolutionalLayer f2(24, 24, 5, 10, 10, 3);
//	FeedForward f3(675, 10);
//	net.add(&f1);	//add them to the network
//	net.add(&f2);
//	net.add(&f3);

//	FeedForward f1(784, 400);
//	FeedForward f2(400, 200);
//	FeedForward f3(200, 10);
//	net.add(&f1);	//add them to the network
//	net.add(&f2);
//	net.add(&f3);

//
	NeuralNetwork net;

	ConvolutionalLayer f1(28, 28,1, 3, 3, 10);	//create FF layers
//	ConvolutionalLayer f2(26, 26,10, 3, 3, 10);	//create FF layers
	FeedForward f3(6760, 10);
	net.add(&f1);	//add them to the network
	//net.add(&f2);
	net.add(&f3);


	data inputs(0);
	data outputs(0);

	std::ifstream is("///home/joseph///Downloads///train.csv");
	generateAndLoad(inputs, outputs, is, 400);

	double test_error = 0;
	test_error = net.test(inputs, outputs);
	std::cout << " test error is initially -- " << test_error << std::endl;

	net.train(inputs, outputs, 10);

//Get the next data sets to use as test set
//	inputs.clear();
//	outputs.clear();
//	generateAndLoad(inputs,outputs,is);
//

	//Test the errir again
	test_error = net.test(inputs, outputs);
	std::cout << " test error post training -- " << test_error << std::endl;
}
