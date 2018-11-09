///*
// * GRU.h
// *
// *  Created on: Sep 20, 2018
// *      Author: joseph
// */
//
//#ifndef BC_INTERNALS_LAYERS_GRU_H_
//#define BC_INTERNALS_LAYERS_GRU_H_
//
//#include <forward_list>
//
//#include "Recurrent_Layer_Base.h"
//#include "Utility.h"
//namespace BC {
//namespace NN {
//
//struct GRU : public Recurrent_Layer_Base {
//
//    using Recurrent_Layer_Base::numb_inputs;
//    using Recurrent_Layer_Base::numb_outputs;
//    using Recurrent_Layer_Base::lr;    //the learning rate
//    using Recurrent_Layer_Base::t;     //current_time_stamp
//    using Recurrent_Layer_Base::max_backprop_length;
////    std::vector<mat_view> x = std::vector<mat_view>(this->get_max_bptt_length());
//
//    cube f, z, c, x;                     //
//    mat dc, df, dz;
//    mat wf, wz;                      //weights
//    vec bf, bz;                      //biases
//
//    mat wfd, wzd;                    //weight deltas
//    vec bfd, bzd;                     //bias deltas
//
//    GRU(int inputs, int outputs) :
//        Recurrent_Layer_Base(inputs, outputs),
//            wf(outputs, outputs), wz(outputs, outputs),
//            bf(outputs), bz(outputs),
//            wfd(outputs, outputs), wzd(outputs, outputs),
//            bfd(outputs), bzd(outputs)
//
//    {
//        wf.randomize(-1, 0);
//        wz.randomize(-1, 1);
//        bf.randomize(-1, 0);
//        bz.randomize(-1, 1);
//    }
//    template<class T>
//    void cache_inputs(const expr::mat<T>& x_) {
//        int i = this->numb_inputs();
//        int o = this->numb_outputs();
//        int b = this->batch_size();
//
//        chunk(x[t], 0, 0)(i, b) =  x_;            //copy the current inputs into the first block of the inputs
//        chunk(x[t], i, 0)(o, o+b) = chunk(x[t-1], i, 0)(o, o+b); //copy the second block (outputs) to the second block
//    }
//
//    template<class T>
//    const auto forward_propagation(const expr::mat<T>& x_) {
//        std::cout << " fp cache " << std::endl;
//        cache_inputs(x_);
//        std::cout << " fp run " << std::endl;
//
//        f[t] = g(wf * x[t] + bf);
//        z[t] = g(wz * x[t] + bz);
//
//        return c[t] = c[t] % f[t] + z[t];
//
//    }
//    template<class T>
//    auto back_propagation(const expr::mat<T>& dy_) {
//        std::cout <<  "d/c " << std::endl;
//        dc = dy_;
//        dz = dc * gd(z[t]);
//        df = dc % c[t - 1] % gd(f[t]);
//
//        return (wf.t() * dz + wz.t() * dz) % gd(x[t]);
//    }
//    void cache_gradients() {
//        wzd -= dz * lr * z[t].t();
//        wfd -= df * lr * f[t].t();
//
//        bzd -= dz;
//        bfd -= df;
//    }
//
//    void update_weights() {
//        wz += wzd;
//        wf += wfd;
//        bz += bzd;
//        bf += bfd;
//
//        wzd.zero();
//        wfd.zero();
//        bzd.zero();
//        bfd.zero();
//    }
//
//    void set_batch_size(int batch_sz) {
//        Recurrent_Layer_Base::set_batch_size(batch_sz);
//        f = cube(numb_outputs(), batch_sz, max_backprop_length);
//        c = cube(numb_outputs(), batch_sz, max_backprop_length);
//        z = cube(numb_outputs(), batch_sz, max_backprop_length);
//        df = mat(numb_outputs(), batch_sz);
//        dc = cube(numb_outputs(), batch_sz);
//        dz = cube(numb_outputs(), batch_sz);
//        x = cube(numb_outputs() + numb_inputs(), batch_sz, max_backprop_length);
//    }
//    void set_max_bptt_length(int len) {
//        Recurrent_Layer_Base::set_max_bptt_length(len);
//        f = cube(this->numb_outputs(), this->batch_size(), len);
//        c = cube(this->numb_outputs(), this->batch_size(), len);
//        z = cube(this->numb_outputs(), this->batch_size(), len);
//        x = cube(this->numb_outputs() + this->numb_inputs(), this->batch_size(), len);
//
//    }
//
//};
//}
//}
//
//
//
//
//#endif /* BC_INTERNALS_LAYERS_GRU_H_ */
