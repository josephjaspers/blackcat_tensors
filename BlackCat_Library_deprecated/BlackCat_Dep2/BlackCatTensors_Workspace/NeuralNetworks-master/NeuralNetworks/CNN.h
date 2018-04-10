/*
#ifndef Filter_jas_h
#define Filter_jas_h
#include "Layer.h"
class CNN : public Layer {
	//CNN Layer --> seperate by channels --> seperate by features 

	CNN* f_next;
	CNN* f_prev;
	void filter_link(CNN* f) {
		f_next = f; 
		f->f_prev = this;
	}

	const unsigned FILTER_INDEX;

	const unsigned LENGTH;					//Input picture length dimension	|| version only accepts square images, length always equals width
	const unsigned WIDTH;					//Input picture width dimension		|| version only accepts square images, width always equals length

	const unsigned STRIDE;					//stride distance

	const unsigned FEATURE_LENGTH;		//feature map length dimension
	const unsigned FEATURE_WIDTH;		//feature map width dimension

	const unsigned POOLED_LENGTH;			//length dimension post pooling
	const unsigned POOLED_WIDTH;			//width dimension post pooling
	const unsigned POOLED_SIZE;

	std::vector<double> bp_max_value; 
	std::vector<int> bp_max_index_x;
	std::vector<int> bp_max_index_y;
	std::vector<Matrix> bpX;
	Matrix w;
	Matrix b;
	Matrix w_gradientStorage;
	Matrix b_gradientStorage;

	Matrix Xt();
	int index_Xt();
	int index_Yt();
public:
	CNN(unsigned input_length, unsigned input_width, unsigned filter_length, unsigned stride);
	CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length, unsigned stride);
	~CNN();
	double findMax(Matrix & img, int & x_store, int & y_store);

	//@Override methods
	Vector forwardPropagation_express(const Vector& x);						//forward propagation [Express does not store activations for BPPT]
	Vector forwardPropagation(const Vector& x);								//forward propagation [Stores activations for BP & BPPT]
	Vector backwardPropagation(const Vector& dy);							//backward propagation[Initial BP]
	Vector backwardPropagation_ThroughTime(const Vector& dy);				//BPPT [Regular BP must be called before BPTT]

	Vector concat_toVector(std::vector<Matrix>& m);

	Vector forwardPropagation_express(const Matrix& x, std::vector<Matrix> collection_outputs);						//forward propagation [Express does not store activations for BPPT]
	Vector forwardPropagation(const Matrix& x, std::vector<Matrix> collection_outputs);								//forward propagation [Stores activations for BP & BPPT]
	Vector backwardPropagation(const Vector& dy, int dy_index);							//backward propagation[Initial BP]
	Vector backwardPropagation_ThroughTime(const Matrix& dy, int dy_index);				//BPPT [Regular BP must be called before BPTT]	

	
	void clearBPStorage();
	void clearGradients();
	void updateGradients();

	//not supported yet
	void write(std::ofstream& os) {};
	void writeClass(std::ofstream& os) {};
	static CNN read(std::ifstream& is) {};

	void printWeights() {
		w.print();
	}
};
#endif
*/