///*
// * ConvLayer.h
// *
// *  Created on: Mar 27, 2018
// *      Author: joseph
// */
//
//#include "Layer.h"
//
//
//#ifndef CONVLAYER_H_
//#define CONVLAYER_H_
//
//#include "Layer.h"
//
//namespace BC {
//namespace NN {
//template<class derived>
//struct Conv : public Layer<derived> {
//
//	using PARAMETER = cube;
//
//	static constexpr int KRNL_DIM = 3;
//
//	scal lr = scal(0.03); //fp_type == floating point
//
//	int filters;
//	int row;
//	int col;
//	int channels;
//
//
//	tensor4 w;
//	omp_unique<tensor4> w_gradientStorage;
//	bp_list<tensor4> ys;
//
//	Conv(std::initializer_list<int> data) :
//			row(data.begin()[0]),
//			col(data.begin()[1]),
//			channels(data.begin()[2]),
//			filters(data.begin()[3]),
//			w(KRNL_DIM,KRNL_DIM,channels, filters)
//	{
//		w.randomize(0, 3);
//	}
//
//
//	template<class T>
//	auto forwardPropagation(const cube& img) {
//		auto y = w.x_corr_stack<2>(img);
//
//		ys().push(cube(y));
//		return this->next().forwardPropagation(ys().first());
//	}
//	template<class T>
//	auto forwardPropagation_Express(const cube& img) const {
//		auto y = w.x_corr_stack<2>(img);
//		return this->next().forwardPropagation(y);
//	}
//
//	template<class T>
//	auto backPropagation(const cube& dy_) {
//
//	}
//
//	void updateWeights() {
//	}
//
//	void clearBPStorage() {
//	}
//
//	void write(std::ofstream& is) {
////		is << INPUTS << ' ';
////		is << OUTPUTS << ' ';
////		w.write(is);
////		b.write(is);
////		x.write(is);
////		y.write(is);
////		dx.write(is);
////		w_gradientStorage.write(is);
////		b_gradientStorage.write(is);
//
//	}
//	void read(std::ifstream& os) {
////		os >> INPUTS;
////		os >> OUTPUTS;
////
////		w.read(os);
////		b.read(os);
////		x.read(os);
////		y.read(os);
////		dx.read(os);
////		w_gradientStorage.read(os);
////		b_gradientStorage.read(os);
//
//	}
//	void setLearningRate(fp_type learning_rate) {
//		lr = learning_rate;
//		this->next().setLearningRate(learning_rate);
//	}
//};
//}
//}


//#endif /* CONVLAYER_H_ */
