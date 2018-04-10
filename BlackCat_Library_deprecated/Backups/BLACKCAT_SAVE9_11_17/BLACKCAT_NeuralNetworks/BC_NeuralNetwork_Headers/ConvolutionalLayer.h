#ifndef BLACKCAT_ConvolutionalLayer_h
#define BLACKCAT_ConvolutionalLayer_h

#include "Layer.h"

class ConvolutionalLayer: public Layer {
public:
	unsigned img_rows;
	unsigned img_cols;
	unsigned img_depth;

	unsigned n_filters;
	unsigned filter_rows;
	unsigned filter_cols;
	unsigned filter_depth;

	unsigned output_rows;
	unsigned output_cols;
	unsigned output_depth;

	unsigned r_positions;
	unsigned c_positions;

	//nonLinearityFunction relu;
	tensor relu(tensor t) {
		for (unsigned i = 0; i < t.size(); ++i) {
			if (t(i) < 0)
				t(i) = 0;
			else if (t(i) > 1) {
				t(i) = 1;
			}
		}
		return t;
	}
	tensor reluD(tensor t) {
		for (unsigned i = 0; i < t.size(); ++i) {
			if (t(i) < 0)
				t(i) = 0;
			else if (t(i) > 1)
				t(i) = 1;
		}
		return t;
	}

	tensor y;					//output
	tensor w;  					//filter
	tensor b;  					//bias
	tensor dx;
	tensor w_gradientStorage;
	tensor b_gradientStorage;
public:

	ConvolutionalLayer(unsigned img_rows, unsigned img_cols, unsigned depth, unsigned filt_rows, unsigned filt_cols, unsigned n_filters) {
		this->img_depth = depth;
		this->img_rows = img_rows;
		this->img_cols = img_cols;

		this->filter_depth = depth;
		this->n_filters = n_filters;
		this->filter_rows = filt_rows;
		this->filter_cols = filt_cols;

		this->output_rows = img_rows - filt_rows + 1;
		this->output_cols = img_cols - filt_cols + 1;
		this->output_depth = n_filters;

		r_positions = img_rows - filter_rows + 1;
		c_positions = img_cols - filter_cols + 1;

		w = tensor(filter_rows, filter_cols, depth, n_filters);
		b = tensor(output_rows, output_cols, n_filters);
		w.randomize(-1, 2);
		b.randomize(-1, 2);

		w_gradientStorage = tensor(filter_rows, filter_cols, depth, n_filters);
		b_gradientStorage = tensor(output_rows, output_cols, n_filters);

		y = tensor(output_rows, output_cols, n_filters);
		dx = tensor(img_rows, img_cols, img_depth);


	}
	virtual vec forwardPropagation(vec data) {
		tensor x = data;
		x.reshape( { img_rows, img_cols, img_depth });
		y = relu(w.x_corr(2, x) + b);

		bpX.store(std::move(x));
		return next ?
				next->forwardPropagation(std::move(y.flatten())) :
				std::move(y.flatten());

	}
	virtual vec forwardPropagation_express(vec data) {
		tensor x = data;
		x.reshape( { img_rows, img_cols, img_depth });

		y = relu(w.x_corr(2, x) + b);

		return next ?
				next->forwardPropagation(std::move(y.flatten())) :
				std::move(y.flatten());
	}
	virtual vec backwardPropagation(vec error) {
		//Shape the error to apropriate dimensions
		tensor dy = error;
		dy.reshape( { output_rows, output_cols, n_filters });
		//get the last inputs
		tensor x(bpX.poll_last());
		//get bias gradient
		b_gradientStorage -= dy;

		//calculate weight gradients
		unsigned e_id = 0;
		for (unsigned f = 0; f < n_filters; ++f) {			//for each filter
			for (unsigned r = 0; r < r_positions; ++r) {	//multiply the subtensor by the value
				for (unsigned c = 0; c < c_positions; ++c) {
					w_gradientStorage[f] -= x( { 0, r, c }, { filter_rows,filter_cols, img_depth }) & dy(e_id);
					++e_id;
					e_id = e_id % dy.size();
				}
			}
		}
		if (prev) {
			tensor dx = {img_rows, img_cols, img_depth};
			dx = 0; //clear that
			x = reluD(x);//derivative that stuff
			e_id = 0;
			for (unsigned f = 0; f < n_filters; ++f) {
				for (unsigned r = 0; r < r_positions; ++r) {
					for (unsigned c = 0; c < c_positions; ++c) {

						dx( { 0, r, c },{ filter_rows, filter_cols, img_depth }) -= w[f].T() & dy(e_id);

						++e_id; e_id = e_id % dy.size();
					}
				}
			}

			return prev->backwardPropagation(dx.flatten());
		} else {
			return error.flatten();
		}
	}
	virtual vec backwardPropagation_ThroughTime(vec dy) {
		return dy;
	}

	//NeuralNetwork update-methods
	virtual void clearBackPropagationStorage() {
		bpX.clear();
	}
	virtual void clearGradientStorage() {
		w_gradientStorage = 0;
		b_gradientStorage = 0;
	}
	virtual void updateGradients() {
		w += w_gradientStorage & lr;
		b += b_gradientStorage & lr; // / (output_rows * output_cols * n_filters);
	}
};
#endif

