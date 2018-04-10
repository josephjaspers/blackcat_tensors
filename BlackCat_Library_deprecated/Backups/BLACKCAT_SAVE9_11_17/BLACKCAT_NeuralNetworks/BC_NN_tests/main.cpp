#include <iostream>
#include "FeedForward.h"
#include <thread>
#include "Layer.h"
#include "NeuralNetwork.h"
#include "RecurrentUnit.h"
#include "LSTM.h"
#include "GatedRecurrentUnit.h"
#include "ConvolutionalLayer.h"

vec ff_one() {
	return vec({0,0,1,0,0,
				0,0,1,0,0,
				0,0,1,0,0,
				0,0,1,0,0,
				0,1,1,1,0,
	});
}
vec ff_two() {
	return vec({0,0,1,0,0,
				0,1,0,1,0,
				0,0,0,1,0,
				0,0,1,0,0,
				0,1,1,1,0,
	});
}
vec ff_three() {
	return vec({0,0,1,1,0,
				0,0,0,1,0,
				0,0,1,1,0,
				0,0,0,1,0,
				0,0,1,1,0,
	});
}
vec ff_four() {
	return vec({0,1,0,1,0,
				0,1,0,1,0,
				0,1,1,1,0,
				0,0,0,1,0,
				0,0,0,1,0,
	});
}

	vec one_1() {return  vec({ 0, 0, 1, 0, 0 }); }
	vec one_2() {return  vec({ 0, 1, 1, 0, 0 }); }
	vec one_3() {return  vec({ 0, 0, 1, 0, 0 }); }
	vec one_4() {return  vec({ 0, 0, 1, 0, 0,}); }
	vec one_5() {return  vec({ 0, 1, 1, 1, 0 }); }

	vec two_1() {return  vec({ 0, 1, 1, 0, 0 }); }
	vec two_2() {return  vec({ 0, 0, 0, 1, 0 }); }
	vec two_3() {return  vec({ 0, 0, 1, 0, 0 }); }
	vec two_4() {return  vec({ 0, 1, 0, 0, 0 }); }
	vec two_5() {return  vec({ 0, 0, 1, 1, 0 }); }

	vec three_1() {return  vec({ 0, 1, 1, 1, 0 }); }
	vec three_2() {return  vec({ 0, 0, 0, 1, 0 }); }
	vec three_3() {return  vec({ 0, 1, 1, 1, 0 }); }
	vec three_4() {return  vec({ 0, 0, 0, 1, 0 }); }
	vec three_5() {return  vec({ 0, 1, 1, 1, 0 }); }

	mat one() {
		tensor m = {5, 5};
		m[0] = one_1();
		m[1] = one_2();
		m[2] = one_3();
		m[3] = one_4();
		m[4] = one_5();
		return m;
	}

	mat two() {
		tensor m = {5, 5};
		m[0] = two_1();
		m[1] = two_2();

		m[2] = two_3();
		m[3] = two_4();
		m[4] = two_5();

		return m;
	}

	mat three() {
		tensor m = {5, 5};
		m[0] = three_1();
		m[1] = three_2();
		m[2] = three_3();
		m[3] = three_4();
		m[4] = three_5();

		return m;
	}

void XOR_Test() {


	std::cout << " main - NN test... " << std::endl;
	vec d1 = { 0, 0 };
	vec d2 = { 1, 1 };
	vec d3 = { 1, 0 };
	vec d4 = { 0, 1 };

	vec o1 = { 1 };
	vec o2 = { 1 };
	vec o3 = { 0 };
	vec o4 = { 0 };

	NeuralNetwork net;
	FeedForward f1(2, 15);
	FeedForward f2(15, 15);
	FeedForward f3(15, 1);

	net.add(&f1);
	net.add(&f2);
	net.add(&f3);

	vec output;
std::cout << " training " << std::endl;
	for (int i = 0; i < 1000; ++i) {
		output = net.forwardPropagation(d1);
		net.backwardPropagation(output - o1);
		net.update();
		output = net.forwardPropagation(d2);
		net.backwardPropagation(output - o2); net.update();
		output = net.forwardPropagation(d3);
		net.backwardPropagation(output - o3); net.update();
		output = net.forwardPropagation(d4);
		net.backwardPropagation(output - o4); net.update();

//		if (i % 100 == 0) {
//			f1.forwardPropagation_express(d1).print();
//			f1.forwardPropagation_express(d2).print();
//			f1.forwardPropagation_express(d3).print();
//			f1.forwardPropagation_express(d4).print();
//		//	int wait;
//			std::cin >> wait;
//		}
	}
std::cout << " post training " << std::endl;
	f1.forwardPropagation_express(d1).print();
	f1.forwardPropagation_express(d2).print();
	f1.forwardPropagation_express(d3).print();
	f1.forwardPropagation_express(d4).print();

	std::cout << " success " << std::endl;
}

void recurrent_numbTrain(NeuralNetwork& net, mat data, vec output) {

	mat prediction;

	for (int i = 0; i < 5; ++i) {
		prediction = net.forwardPropagation(data[i]);
	}
	net.backwardPropagation(prediction - output);

	for (int i = 0; i < 4; ++i) {
		net.backwardPropagation_ThroughTime(vec(output.size()));
	}
	net.update();
}
vec recurrent_numbFP(NeuralNetwork& net, mat data) {
	vec pred;
	for (int i = 0; i < 5; ++i) {
		pred = net.forwardPropagation(data[i].T());
	}
	return pred;
}

void recurrent_numbTest() {

	std::cout << " main - NN test... " << std::endl;
	mat d1 = one();
	mat d2 = two();
	mat d3 = three();

	vec o1 = {1,0,0};
	vec o2 = {0,1,0};
	vec o3 = {0,0,1};

std::cout << " gen network " << std::endl;
	NeuralNetwork net;

	GatedRecurrentUnit f1(5, 20);
	GatedRecurrentUnit f2(20, 10);
	GatedRecurrentUnit f3(10, 3);

	net.add(&f1);
	net.add(&f2);
	net.add(&f3);

	vec output;
std::cout << " training " << std::endl;
	for (int i = 0; i < 2000; ++i) {

		recurrent_numbTrain(net, (d1), o1);
		recurrent_numbTrain(net, (d2), o2);
		recurrent_numbTrain(net, (d3), o3);
	}
std::cout << " post training " << std::endl;

	recurrent_numbFP(net, one()).print();
	recurrent_numbFP(net, two()).print();
	recurrent_numbFP(net, three()).print();

	std::cout << " success " << std::endl;
}

void ff_train(NeuralNetwork& net,  vec data,  vec output) {
	auto o = net.forwardPropagation(data);
	net.backwardPropagation(o - output);
	net.update();
}
vec ff_fp(NeuralNetwork& net, const vec data) {
	return net.forwardPropagation_express(data);
}
void ff_trainI(int iter, NeuralNetwork& net, vec data, vec output) {
	for (int i = 0; i < iter; ++i){
		ff_train(net, data, output);
	}
}

void feedforward_numbTest() {

	std::cout << " main - NN test... " << std::endl;
	vec d1 = ff_one();
	vec d2 = ff_two(); //= two().flatten();
	vec d3 = ff_three();//= three().flatten();
	vec d4 = ff_four();
	vec o1 = {1,0,0,0};
	vec o2 = {0,1,0,0};
	vec o3 = {0,0,1,0};
	vec o4 = {0,0,0,1};

std::cout << " gen network " << std::endl;
	NeuralNetwork net;

//	ConvolutionalLayer f1(5, 5, 1, 3, 3, 10);
//	ConvolutionalLayer f2(3, 3, 10, 2, 2, 4);
//	FeedForward f3(160, 4);
//	net.add(&f1);
//	net.add(&f2);
//	net.add(&f3);

	ConvolutionalLayer f1(5, 5, 1, 3, 3, 10);
	FeedForward f3(90, 4);

	net.add(&f1);
	net.add(&f3);

	std::cout << "linking " << std::endl;

	vec output;
std::cout << " training " << std::endl;

	for (int i = 0; i < 1000; ++i) {
		ff_train(net, d1, o1);
		ff_train(net, d2, o2);
		ff_train(net, d3, o3);
		ff_train(net, d4, o4);

	}


std::cout << " post training " << std::endl;

	ff_fp(net,ff_one()).print();
	ff_fp(net, ff_two()).print();
	ff_fp(net, ff_three()).print();
	ff_fp(net, ff_four()).print();


	std::cout << " success " << std::endl;
}
int mainTests() {




	//XOR_Test();
	feedforward_numbTest();
	//recurrent_numbTest();
	return 0;

}

