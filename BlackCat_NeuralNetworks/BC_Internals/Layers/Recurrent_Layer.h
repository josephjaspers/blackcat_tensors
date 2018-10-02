///*
// * Recurrent_Standard_Layer.cu
// *
// *  Created on: Jan 28, 2018
// *      Author: joseph
// */
//
//#ifndef Recurrent_Standard_Layer_CU_
//#define Recurrent_Standard_Layer_CU_
//
//#include "Recurrent_Layer_Base.h"
//#include <vector>
//
//namespace BC {
//namespace NN {
//
//struct Recurrent_Standard_Layer : public Recurrent_Layer_Base {
//
//	using Recurrent_Layer_Base::lr;	//the learning rate
//	using Recurrent_Layer_Base::t;
//
//	mat dy;          //error
//	cube y;           //outputs
//	std::vector<mat_view> x = std::vector<mat_view>(this->get_max_bptt_length());             //inputs
//
//	mat w;                  //weights
//	mat r; 					//recurrent_weights
//	vec b;                  //biases
//	mat wd;					//weight delta
//	mat rd;
//	vec bd;					//bias delta
//
//
//	Recurrent_Standard_Layer(int inputs, int outputs) :
//		Recurrent_Layer_Base(inputs, outputs),
//			w(outputs, inputs),
//			r(outputs, outputs),
//			b(outputs),
//			wd(outputs, inputs),
//			rd(outputs, outputs),
//			bd(outputs)
//	{
//		w.randomize(-1, 1);
//		b.randomize(-1, 1);
//		r.randomize(-1, 1);
//	}
//	template<class T>
//	const auto forward_propagation(const expr::mat<T>& x_) {
//		x[t] = mat_view(x_);
//		 y[t] = g(w * x[t] + r * y[t-1] + b);
//		 t++;
//		 return y[t];
//	}
//	template<class T>
//	auto back_propagation(const expr::mat<T>& dy_) {
//		dy = dy_;
//		return (w.t() * dy) % gd(x[t]);
//	}
//	void update_weights() {
//		r -= rd;
//		w += wd;
//		b += bd;
//		wd.zero();
//		bd.zero();
//		rd.zero();
//		dy.zero();
//	}
//	void cache_gradients() {
//		wd -= dy * lr * x[t].t();
//		rd -= dy * lr * y[t-1].t();
//		bd -= dy * lr;
//	}
//
//	void set_batch_size(int batch_sz) {
//		y = cube(this->numb_outputs(), batch_sz, this->get_max_bptt_length());
//		dy = mat(this->numb_outputs(), batch_sz);
//	}
//
//	auto& inputs()  { return x; }
//	auto& outputs() { return y; }
//	auto& deltas()  { return dy;}
//	auto& weights()	{ return w; }
//	auto& bias()	{ return b; }
//
//
//};
//using Rec = Recurrent_Standard_Layer;
//}
//}
//
//#endif /* Recurrent_Standard_Layer_CU_ */
