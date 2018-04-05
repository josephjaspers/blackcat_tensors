/*
 * ConvLayer.h
 *
 *  Created on: Mar 27, 2018
 *      Author: joseph
 */

#ifndef CONVLAYER_H_
#define CONVLAYER_H_

#include "Layer.h"

namespace BC {

template<class derived>
struct Conv : public Layer<derived> {


public:
	 scal lr = scal(0.03); //fp_type == floating point

	 int filters;
	 int row;
	 int col;
	 int channels;

	tensor4 w_gradientStorage;
	tensor4 w;

	cube y;
	cube dy;
	cube x;
	cube dx;

	vec flat;


	static constexpr int KRNL_DIM = 3;

	//rows,cols,channels, numbe_filters,
	Conv(std::tuple<int,int,int, int> data) :
			row(std::get<0>(data)),
			col(std::get<1>(data)),
			channels(std::get<2>(data)),
			filters(std::get<3>(data)),

			x(row, col, channels),
			y(row + KRNL_DIM - 1, col + KRNL_DIM - 1, filters),

			w_gradientStorage(KRNL_DIM, KRNL_DIM, channels, filters),
			w(KRNL_DIM, KRNL_DIM, channels, filters),

			dx(row, col, channels),
			dy(row + KRNL_DIM - 1, col + KRNL_DIM - 1, filters),
			Layer<derived>(row * col * channels),

			flat(0)
	{
		w.randomize(0, 2);
		w_gradientStorage.zero();
	}


	template<class T>
	auto forwardPropagation(const _vec<T>& x_) {
		x = x_;

		for (int f = 0; f < filters; ++f) {
			for (int c = 0; c < channels; ++c) {
				y[f] = w[c][f].x_corr_padded(x[c]);
				y[f].print();

			}
		}
		y.print();
		flat = y;
		std::cout << "--------------------------------------" << std::endl;
		return this->next().forwardPropagation(flat);
	}
	template<class T>
	auto forwardPropagation_Express(const _vec<T>& x_) const {
		x = x_;

		for (int f = 0; f < filters; ++f) {
			for (int c = 0; c < channels; ++c) {
				y[f] = w[c][f].x_corr_padded(x[c]);
			}
		}
		return this->next().forwardPropagation(flat);
	}

	template<class T>
	auto backPropagation(const _vec<T>& dy_) {
		dy = dy_;
		for (int f = 0; f < filters; ++f) {
			for (int c = 0; c < channels; ++c) {
				w_gradientStorage[c][f] -= x[c].x_corr(dy[f]);
				dx[c] += w[c][f].x_corr(dy[f]);

			}
		}

		flat = dx;
		return this->prev().backPropagation(flat);

	}

	void updateWeights() {
		w += w_gradientStorage * lr;
		this->next().updateWeights();
	}

	void clearBPStorage() {
		w_gradientStorage.zero();
		this->next().clearBPStorage();
	}

	void write(std::ofstream& is) {
//		is << INPUTS << ' ';
//		is << OUTPUTS << ' ';
//		w.write(is);
//		b.write(is);
//		x.write(is);
//		y.write(is);
//		dx.write(is);
//		w_gradientStorage.write(is);
//		b_gradientStorage.write(is);

	}
	void read(std::ifstream& os) {
//		os >> INPUTS;
//		os >> OUTPUTS;
//
//		w.read(os);
//		b.read(os);
//		x.read(os);
//		y.read(os);
//		dx.read(os);
//		w_gradientStorage.read(os);
//		b_gradientStorage.read(os);

	}
	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}
};
}


#endif /* CONVLAYER_H_ */
