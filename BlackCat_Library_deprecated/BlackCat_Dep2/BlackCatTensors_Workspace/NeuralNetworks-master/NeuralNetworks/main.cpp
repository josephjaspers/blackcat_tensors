#include "stdafx.h"
#include "FeedForward.h"
#include "GRU.h"
#include "CNN.h"
#include "RecurrentUnit.h"
#include "NeuralNetwork.h"
#include "LSTM.h"
using namespace std;

namespace testClass {
	void print(vector<double> v) {
		cout.precision(1);
		for (double dz : v) {
			cout << dz << " ";
		}
		cout << endl;
	}
	void printConf(vector<double> v) {
		cout.precision(2);
		int index = -1;
		double best = 0;
		for (int i = 0; i < v.size(); ++i) {
			if (v[i] > best) {
				best = v[i];
				index = i;
			}
		}
		cout << "(" << index << ")" << " conf (" << best << ")" << endl << endl;
	}

	vector<vector<double>> get_recurrent_Zero() {
		vector<double> r1 = { 0,0,1,0,0 };
		vector<double> r2 = { 0,1,0,1,0 };
		vector<double> r3 = { 0,1,0,1,0 };
		vector<double> r4 = { 0,1,0,1,0 };
		vector<double> r5 = { 0,0,1,0,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_One() {
		vector<double> r1 = { 0,1,1,0,0 };
		vector<double> r2 = { 0,0,1,0,0 };
		vector<double> r3 = { 0,0,1,0,0 };
		vector<double> r4 = { 0,0,1,0,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_Two() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,0,0,1,0 };
		vector<double> r3 = { 0,0,1,0,0 };
		vector<double> r4 = { 0,1,0,0,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_Three() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,0,0,1,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,0,0,1,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_Four() {
		vector<double> r1 = { 0,1,0,1,0 };
		vector<double> r2 = { 0,1,0,1,0 };
		vector<double> r3 = { 0,1,1,1,1 };
		vector<double> r4 = { 0,0,0,1,0 };
		vector<double> r5 = { 0,0,0,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_Five() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,1,0,0,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,0,0,1,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_Six() {
		vector<double> r1 = { 0,1,0,0,0 };
		vector<double> r2 = { 0,1,0,0,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,1,0,1,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_Seven() {
		vector<double> r1 = { 0,1,1,0,0 };
		vector<double> r2 = { 0,0,0,1,0 };
		vector<double> r3 = { 0,0,1,1,0 };
		vector<double> r4 = { 0,1,0,0,0 };
		vector<double> r5 = { 0,1,0,0,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_Eight() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,1,0,1,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,1,0,1,0 };
		vector<double> r5 = { 0,1,1,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}
	vector<vector<double>> get_recurrent_Nine() {
		vector<double> r1 = { 0,1,1,1,0 };
		vector<double> r2 = { 0,1,0,1,0 };
		vector<double> r3 = { 0,1,1,1,0 };
		vector<double> r4 = { 0,0,0,1,0 };
		vector<double> r5 = { 0,0,0,1,0 };

		vector<vector<double>> n = { r1,r2,r3,r4,r5 };
		return n;
	}

	vector<double> getZero() {
		vector<double> n =
		{ 0,0,1,0,0,
			0,1,0,1,0,
			0,1,0,1,0,
			0,1,0,1,0,
			0,0,1,0,0, };
		return n;
	}
	vector<double> getOne() {
		vector<double> n =
		{ 0,1,1,0,0,
			0,0,1,0,0,
			0,0,1,0,0,
			0,0,1,0,0,
			0,1,1,1,0, };
		return n;
	}
	vector<double> getTwo() {
		vector<double> n =
		{ 0,1,1,1,0,
			0,0,0,1,0,
			0,0,1,0,0,
			0,1,0,0,0,
			0,1,1,1,0, };
		return n;
	}
	vector<double> getThree() {
		vector<double> n =
		{ 0,1,1,1,0,
			0,0,0,1,0,
			0,1,1,1,0,
			0,0,0,1,0,
			0,1,1,1,0, };
		return n;
	}
	vector<double> getFour() {
		vector<double> n =
		{ 0,1,0,1,0,
			0,1,0,1,0,
			0,1,1,1,1,
			0,0,0,1,0,
			0,0,0,1,0, };
		return n;
	}
	vector<double> getFive() {
		vector<double> n =
		{ 0,1,1,1,0,
			0,1,0,0,0,
			0,1,1,1,0,
			0,0,0,1,0,
			0,1,1,1,0, };
		return n;
	}
	vector<double> getSix() {
		vector<double> n =
		{ 0,1,1,1,0,
			0,1,0,0,0,
			0,1,1,1,0,
			0,1,0,1,0,
			0,1,1,1,0, };
		return n;
	}
	vector<double> getSeven() {
		vector<double> n =
		{ 0,1,1,1,0,
			0,0,0,1,0,
			0,0,1,0,0,
			0,1,0,0,0,
			0,1,0,0,0, };
		return n;
	}
	vector<double> getEight() {
		vector<double> n =
		{ 0,1,1,1,0,
			0,1,0,1,0,
			0,1,1,1,0,
			0,1,0,1,0,
			0,1,1,1,0, };
		return n;
	}
	vector<double> getNine() {
		vector<double> n =
		{ 0,1,1,1,0,
			0,1,0,1,0,
			0,1,1,1,0,
			0,0,0,1,0,
			0,0,0,1,0, };
		return n;
	}
}

using namespace testClass;

void printConf(Vector& x) {
	x.print();

	int best = -1;
	double bestVal = -1;
	for (int i = 0; i < x.size(); ++i) {
		if (bestVal < x[i]) {
			bestVal = x[i];
			best = i;
		}
	}
	cout << "~~ (" << best << ") " << " conf: " << bestVal;
}

void reccurent_drTest() {
	NeuralNetwork network;
	ifstream is("testFile");

	network.push_back(new LSTM(5, 10));
	//network.push_back(new GRU(10, 15));
	//network.push_back(new RecurrentUnit(15, 20));
	network.push_back(new FeedForward(10, 10));
	//network.read(is);

	int train = 1;
	while (train > 0) {
		cout.precision(1);
		cout << " testing 0 " << endl;
		Vector t0 = network.predict(get_recurrent_Zero());
		printConf(t0);
		cout << endl << " testing 1 " << endl;
		Vector t1 = network.predict(get_recurrent_One());
		printConf(t1);

		cout << endl << " testing 2 " << endl;
		Vector t2 = network.predict(get_recurrent_Two());
		printConf(t2);

		cout << endl << " testing 3 " << endl;
		Vector t3 = network.predict(get_recurrent_Three());
		printConf(t3);

		cout << endl << " testing 4 " << endl;
		Vector t4 = network.predict(get_recurrent_Four());
		printConf(t4);

		cout << endl << " testing 5 " << endl;
		Vector t5 = network.predict(get_recurrent_Five());
		printConf(t5);

		cout << endl << " testing 6 " << endl;
		Vector t6 = network.predict(get_recurrent_Six());
		printConf(t6);

		cout << endl << " testing 7 " << endl;
		Vector t7 = network.predict(get_recurrent_Seven());
		printConf(t7);

		cout << endl << " testing 8 " << endl;
		Vector t8 = network.predict(get_recurrent_Eight());
		printConf(t8);

		cout << endl << " testing 9" << endl;
		Vector t9 = network.predict(get_recurrent_Nine());
		printConf(t9);

		cout << endl;

		cout << " input training iterations " << endl;
		cin >> train;
		for (int i = 0; i < train; ++i) {
			network.train(get_recurrent_Zero(), vector<double> {1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
			network.train(get_recurrent_One(), vector<double>  {0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
			network.train(get_recurrent_Two(), vector<double>  {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
			network.train(get_recurrent_Three(), vector<double>{0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
			network.train(get_recurrent_Four(), vector<double> {0, 0, 0, 0, 1, 0, 0, 0, 0, 0});
			network.train(get_recurrent_Five(), vector<double> {0, 0, 0, 0, 0, 1, 0, 0, 0, 0});
			network.train(get_recurrent_Six(), vector<double>  {0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
			network.train(get_recurrent_Seven(), vector<double>{0, 0, 0, 0, 0, 0, 0, 1, 0, 0});
			network.train(get_recurrent_Eight(), vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
			network.train(get_recurrent_Nine(), vector<double> {0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
		}
	}

	//ofstream os("testFile");
	//network.write(os);
	//os.close();
}
void conv_drTest() {


	NeuralNetwork network = NeuralNetwork();

	//network.push_back(new FeedForward(25, 320));
	//network.push_back(new FeedForward(320, 10));
	network.push_back(new FF_norec(25, 10));
	network.push_back(new FF_norec(10, 10));

	//network.push_back(new CNN(5, 5, 2, 1));
	//network.push_back(new FeedForward(16, 10));


	ifstream is("t_file");
	if (is.is_open()) {
		cout << " file sucessfully open" << endl;
	}
	else {
		cout << "file open fail " << endl;
	}
	//network.read(is);
	is.close();

	int train = 1;
	while (train > 0) {
		cout.precision(1);
		cout << " testing 0 " << endl;
		Vector t0 = network.predict(getZero());
		printConf(t0);
		cout << endl << " testing 1 " << endl;
		Vector t1 = network.predict(getOne());
		printConf(t1);

		cout << endl << " testing 2 " << endl;
		Vector t2 = network.predict(getTwo());
		printConf(t2);

		cout << endl << " testing 3 " << endl;
		Vector t3 = network.predict(getThree());
		printConf(t3);

		cout << endl << " testing 4 " << endl;
		Vector t4 = network.predict(getFour());
		printConf(t4);

		cout << endl << " testing 5 " << endl;
		Vector t5 = network.predict(getFive());
		printConf(t5);

		cout << endl << " testing 6 " << endl;
		Vector t6 = network.predict(getSix());
		printConf(t6);

		cout << endl << " testing 7 " << endl;
		Vector t7 = network.predict(getSeven());
		printConf(t7);

		cout << endl << " testing 8 " << endl;
		Vector t8 = network.predict(getEight());
		printConf(t8);

		cout << endl << " testing 9" << endl;
		Vector t9 = network.predict(getNine());
		printConf(t9);

		cout << endl;

		cout << " input training iterations " << endl;
		cin >> train;
		for (int i = 0; i < train; ++i) {
			network.train(getZero(), vector<double> {1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
			network.train(getOne(), vector<double>  {0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
			network.train(getTwo(), vector<double>  {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
			network.train(getThree(), vector<double>{0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
			network.train(getFour(), vector<double> {0, 0, 0, 0, 1, 0, 0, 0, 0, 0});
			network.train(getFive(), vector<double> {0, 0, 0, 0, 0, 1, 0, 0, 0, 0});
			network.train(getSix(), vector<double>  {0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
			network.train(getSeven(), vector<double>{0, 0, 0, 0, 0, 0, 0, 1, 0, 0});
			network.train(getEight(), vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
			network.train(getNine(), vector<double> {0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
		}
	}

	ofstream os("t_file");
	if (os.is_open()) {
		cout << " open t_file " << endl;
	}
	network.write(os);
	os.close();
}
void XORtest() {
	//initialize training set 
	Vector i1(std::vector<double> {0, 0});
	Vector i2(std::vector<double> {1, 1});
	Vector i3(std::vector<double> {1, 0});
	Vector i4(std::vector<double> {0, 1});
	Vector o1(std::vector<double> {1});
	Vector o2(std::vector<double> {1});
	Vector o3(std::vector<double> {0});
	Vector o4(std::vector<double> {0});
	//initialize network
	NeuralNetwork network;
	network.push_back(new FeedForward(2, 12)); //generate a feedforward layer with 2 inputs 5 outputs
	network.push_back(new FeedForward(12, 10));
	network.push_back(new FeedForward(10, 1)); //generate a feedforwad layer with 5 inputs 1 outputs 

	int train = 1;
	while (train > 0) {

		cout << " testing 1, 1 " << endl;
		network.predict(i1).print();
		cout << endl << " testing 0, 0 " << endl;
		network.predict(i2).print();
		cout << endl << " testing 1, 0 " << endl;
		network.predict(i3).print();
		cout << endl << " testing 0, 1 " << endl;
		network.predict(i4).print();
		cout << endl;

		cout << " input training iterations " << endl;
		cin >> train;
		for (int i = 0; i < train; ++i) {
			network.train(i1, o1);
			network.train(i2, o2);
			network.train(i3, o3);
			network.train(i4, o4);
		}
	}
}

int main() {
 conv_drTest();
 reccurent_drTest();
}
