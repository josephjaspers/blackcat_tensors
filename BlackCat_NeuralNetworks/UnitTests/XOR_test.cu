#include "../BlackCat_NeuralNetworks.h"

namespace BC {


void XOR_Test() {

	std::cout << " main - NN test... " << std::endl;
	vec d1 = { 0, 0 };
	vec d2 = { 1, 1 };
	vec d3 = { 1, 0 };
	vec d4 = { 0, 1 };

	vec o1 = { 0 };
	vec o2 = { 0 };
	vec o3 = { 1 };
	vec o4 = { 1 };

	FeedForward f1(2, 40);
	FeedForward f3(40, 1);

//	NeuralNetwork<FeedForward,FeedForward> net(f1, f3);
	auto net = generateNetwork(f1, f3);
	vec output(1);
std::cout << " training " << std::endl;
	for (int i = 0; i < 1000; ++i) {
		output = net.forwardPropagation(d1);
		vec res = output - o1;

		net.backPropagation(res);
		net.updateWeights();
		net.clearBPStorage();

		output = net.forwardPropagation(d2);
		res = output - o2;
		net.backPropagation(res);
		net.updateWeights();
		net.clearBPStorage();


		output = net.forwardPropagation(d3);
		res = output - o3;
		net.backPropagation(res);
		net.updateWeights();
		net.clearBPStorage();

		output = net.forwardPropagation(d4);
		res = output - o4;
		net.backPropagation(res);
		net.updateWeights();
		net.clearBPStorage();

	}
std::cout << " post training " << std::endl;


	vec o11 = net.forwardPropagation(d1); o11.print();
	vec o22 = net.forwardPropagation(d2); o22.print();
	vec o33 = net.forwardPropagation(d3); o33.print();
	vec o44 = net.forwardPropagation(d4); o44.print();

	std::cout << " success " << std::endl;
}

//#define DEFAULT_DEVICE_GPU
//#define BLACKCAT_DEFAULT_MATHLIB_GPU


}//end namespace

//int main() {
//	BC::XOR_Test();
//	std::cout << "success " << std::endl;
//	return 0;
//}

