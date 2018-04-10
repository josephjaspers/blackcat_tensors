#include "stdafx.h"
#include "CNN.h"
/*
Matrix CNN::Xt()
{
	if (bpX.empty()) {
		return Matrix(LENGTH, WIDTH);
	}
	else {
		return bpX.back();
	}
}

int CNN::index_Xt()
{
	if (bp_max_index_x.empty()) {
		return -1;
	}
	else {
		return bp_max_index_x.back();
	}
}

int CNN::index_Yt()
{
	if (bp_max_index_y.empty()) {
		return -1;
	}
	else {
		return bp_max_index_y.back();
	}
}

CNN::CNN(unsigned input_length, unsigned input_width, unsigned filter_length, unsigned stride) :
	LENGTH(input_length), WIDTH(input_width),
	FEATURE_LENGTH(filter_length), FEATURE_WIDTH(filter_length),
	STRIDE(stride),
	POOLED_LENGTH((floor(input_length - filter_length) / stride) + 1),		//no automatic padding of 0s
	POOLED_WIDTH((floor(input_length - filter_length) / stride) + 1),		//no automatic padding of 0s 
	POOLED_SIZE(POOLED_LENGTH * POOLED_WIDTH),
	FILTER_INDEX(1),
	Layer(input_length * input_width, ((floor(input_length - filter_length / stride) + 1) * (floor(input_width - filter_length) / 1) + 1))
{
	w = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
	b = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);

	w_gradientStorage = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
	b_gradientStorage = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
		
	bp_max_index_x = std::vector<int>(POOLED_LENGTH * POOLED_WIDTH);
	bp_max_index_x = std::vector<int>(POOLED_LENGTH * POOLED_WIDTH);

	randomize(w, -4, 4);
	randomize(b, -4, 4);


	//g.setNonLinearityFunction(0);
}
CNN::CNN(unsigned input_length, unsigned input_width, unsigned numb_filters, unsigned filter_length, unsigned stride) :
LENGTH(input_length), WIDTH(input_width),
FEATURE_LENGTH(filter_length), FEATURE_WIDTH(filter_length),
STRIDE(stride),
POOLED_LENGTH((floor(input_length - filter_length) / stride) + 1),		//no automatic padding of 0s
POOLED_WIDTH((floor(input_length - filter_length) / stride) + 1),		//no automatic padding of 0s 
POOLED_SIZE(POOLED_LENGTH * POOLED_WIDTH),
FILTER_INDEX(numb_filters - 1),
Layer(input_length * input_width, ((floor(input_length - filter_length / stride) + 1) * (floor(input_width - filter_length) / 1) + 1))
{
	w = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
	b = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);

	w_gradientStorage = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
	b_gradientStorage = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);

	bp_max_index_x = std::vector<int>(POOLED_LENGTH * POOLED_WIDTH);
	bp_max_index_x = std::vector<int>(POOLED_LENGTH * POOLED_WIDTH);

	randomize(w, -4, 4);
	randomize(b, -4, 4);

	if (FILTER_INDEX > 0) {
		CNN* output = new CNN(LENGTH, WIDTH, numb_filters - 1, filter_length, stride);
		filter_link(output);
	} 
}
CNN::~CNN()
{
	if (f_next != nullptr) {
		delete f_next;
	}
}
double CNN::findMax(Matrix& img, int& x_store, int& y_store) {
	double max = img[0][0];
	x_store = 0;
	y_store = 0;
	for (int x = 0; x < img.length(); ++x) {
		for (int y = 0; y < img.width(); ++y) {
			if (max < img[x][y]) {
				max = img[x][y];
				x_store = x;
				y_store = y;
			}
		}
	}
	return max;
}
//doesnt work????
Vector CNN::forwardPropagation_express(const Vector &input)
{
	Matrix img1 = vec_toMatrix(input, LENGTH, WIDTH);
	std::vector<Matrix> outputs = std::vector<Matrix>();
	outputs.reserve(FILTER_INDEX + 1);
	Vector v_output = forwardPropagation_express(img1, outputs);

	if (next != nullptr) {
		return next->forwardPropagation_express(v_output);
	}
	else {
		return v_output;
	}
}
Vector CNN::forwardPropagation(const Vector & input)
{
	Matrix img_chk = vec_toMatrix(input, LENGTH, WIDTH);
	std::vector<Matrix> outputs = std::vector<Matrix>();
	outputs.reserve(FILTER_INDEX + 1);
	
	//img_chk.print();
	Vector v_output = forwardPropagation(img_chk, outputs);
	//v_output.print();
	if (next != nullptr) {
		return next->forwardPropagation(v_output);
	}
	else {
		return v_output;
	}

}
static double sigmoid(double d) {
	return 1 / (1 + pow(2.71828, -d));
}
static double sigmoid_deriv(double d) {
	return d *= (1 - d);
}
Vector CNN::backwardPropagation(const Vector & dy)
{
	return backwardPropagation(dy, 0);

}

Vector CNN::backwardPropagation_ThroughTime(const Vector & dy)
{
	std::cout << " BPTT attempt FAILED  __ I HAVENT WRITTEN THIS YET SORRY" << std::endl;
	throw std::invalid_argument("not supported");
}
Vector CNN::concat_toVector(std::vector<Matrix>& stack_matrix) {
	int sz = 0;
	for (Matrix m : stack_matrix) {
		sz += m.size();
	}

	Vector r = Vector(sz);

	int v_index = 0;
	for (int i = 0; i < stack_matrix.size(); ++i) {
		for (int x = 0; x < stack_matrix[i].length(); ++x) {
			for (int y = 0; y < stack_matrix[i].width(); ++y) {
				r[v_index] = stack_matrix[i][x][y];
				v_index++;
			//	std::cout << " r[v] = " << r[v_index] << " stack matix [i][x][y] = " << stack_matrix[i][x][y] << std::endl;
			}
		}
	}
	//r.print();

	return r;
}
Vector CNN::forwardPropagation_express(const Matrix & img, std::vector<Matrix> collection_outputs)
{	Matrix pooled = Matrix(POOLED_LENGTH, POOLED_WIDTH);

	for (int x = 0; x < img.length() - FEATURE_LENGTH + 1; x += STRIDE) {
		for (int y = 0; y < img.width() - FEATURE_WIDTH + 1; y += STRIDE) {
			Matrix a = Matrix(FEATURE_LENGTH, FEATURE_WIDTH);
			a = img.sub_Matrix(x, y, FEATURE_LENGTH, FEATURE_WIDTH);
			Matrix conv = (w * a + b);
			g(conv);

			pooled[x / STRIDE][y / STRIDE] = conv.max();
		}
	}
	collection_outputs.push_back(pooled);


	if (f_next != nullptr) {
		return f_next->forwardPropagation_express(img, collection_outputs);
	}
	else {
		return concat_toVector(collection_outputs);
	}
}
Vector CNN::forwardPropagation(const Matrix& img, std::vector<Matrix> collection_outputs)
{
	bp_max_index_x.clear();
	bp_max_index_y.clear();

	Matrix pooled = Matrix(POOLED_LENGTH, POOLED_WIDTH);
	bpX.push_back(img);

	for (int x = 0; x < img.length() - FEATURE_LENGTH + 1; x += STRIDE) {
		for (int y = 0; y < img.width() - FEATURE_WIDTH + 1; y += STRIDE) {
			Matrix a = img.sub_Matrix(x, y, FEATURE_LENGTH, FEATURE_WIDTH);

			Matrix conv = g(w * a + b);

			int max_index_x;
			int max_index_y;
			pooled[x / STRIDE][y / STRIDE] = findMax(conv, max_index_x, max_index_y);

			bp_max_value.push_back(pooled[x / STRIDE][y / STRIDE]);
			bp_max_index_x.push_back(max_index_x);
			bp_max_index_y.push_back(max_index_y);
		}
	}

	collection_outputs.push_back(pooled);
	if (f_next != nullptr) {
		return f_next->forwardPropagation(img, collection_outputs);
	}
	else {
		//concat_toVector(collection_outputs).print();
		return concat_toVector(collection_outputs);
	}
}
Vector CNN::backwardPropagation(const Vector & dy, int dy_base_index)
{
	//Vector dy = dy_col.sub_Vector(dy_base_index, POOLED_SIZE);

	Matrix& img_xt = Xt();

	int dy_index = 0;
	for (int x = 0; x < img_xt.length() - FEATURE_LENGTH; x += STRIDE) {
		for (int y = 0; y < img_xt.width() - FEATURE_WIDTH; y += STRIDE) {
			//delta[bp_max_index_x[dy_index]][bp_max_index_y[dy_index]] = dy.get(dy_index);
			//w_gradientStorage -= delta; &g.d(g(img_xt.sub_Matrix(x, y, FEATURE_LENGTH, FEATURE_WIDTH)));
			//b_gradientStorage -= delta;
			w_gradientStorage[bp_max_index_x[dy_index]][bp_max_index_y[dy_index]] -= dy.get(dy_index + dy_base_index) * sigmoid_deriv(bp_max_value[dy_index]);
			b_gradientStorage[bp_max_index_x[dy_index]][bp_max_index_y[dy_index]] -= dy.get(dy_index + dy_base_index);

			dy_index++;
		}
	}
	bpX.pop_back();

	if (f_next != nullptr) {
		return f_next->backwardPropagation(dy, (dy_base_index + POOLED_SIZE));
	}
	else {
		return dy;
	}
}
Vector CNN::backwardPropagation_ThroughTime(const Matrix & dy, int dy_index)
{
	std::cout << " not supported " << std::endl;
	throw std::invalid_argument("error");
}
void CNN::clearBPStorage()
{
	bpX.clear();

	if (f_next != nullptr) {
		f_next->clearBPStorage();
	}

	Layer::clearBPStorage();

}

void CNN::clearGradients()
{

	w_gradientStorage.fill(0);
	b_gradientStorage.fill(0);

	if (f_next != nullptr) {
		f_next->clearGradients();
	}

	Layer::clearGradients();

}

void CNN::updateGradients()
{
	w += w_gradientStorage & lr;
	b += b_gradientStorage & lr;

	if (f_next != nullptr) {
		f_next->updateGradients();
	}

	Layer::updateGradients();

}
*/