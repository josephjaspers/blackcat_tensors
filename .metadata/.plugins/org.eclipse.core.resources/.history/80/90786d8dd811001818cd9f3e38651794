#include "../BlackCat_NeuralNetworks.h"

namespace BC {


void XOR_Test() {

	std::cout << " main - NN test... " << std::endl;
	vec d1(2); d1[0] = 0; d1[1] =1;
	vec d2 = { 1, 1 };
	vec d3 = { 1, 0 };
	vec d4 = { 0, 1 };

	vec o1 = { 0 };
	vec o2 = { 0 };
	vec o3 = { 1 };
	vec o4 = { 1 };

	FeedForward f1(2, 40);
	FeedForward f2(40, 1);
	OutputLayer o_(1);

//	NeuralNetwork<FeedForward,FeedForward> net(f1, f3);
	auto net = generateNetwork(f1, f2, o_);
std::cout << " training " << std::endl;
	for (int i = 0; i < 1000; ++i) {
		net.forwardPropagation(d1);
		net.backPropagation(o1);
		net.updateWeights();
		net.clearBPStorage();

		net.forwardPropagation(d1);
		net.backPropagation(o1);
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

