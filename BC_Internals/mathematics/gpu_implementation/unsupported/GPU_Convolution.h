/*
 * GPU_Convolution.h
 *
 *  Created on: Jul 5, 2018
 *      Author: joseph
 */

#ifndef GPU_CONVOLUTION_H_
#define GPU_CONVOLUTION_H_

#include "GPU_Convolution_impl.cu"

namespace BC {

template<class core_lib>
struct GPU_Convolution {

    static int blocks(int sz) { return core_lib::blocks(sz); }
    static int threads() { return core_lib::threads(); }


    template<class array_out, class array_in>
    static void img2col_2d(array_out out, array_in in) {
        BC::gpu_impl::img2col_2d<<<blocks(out.size()), threads()>>>(out, in);
    }
    template<class array_out, class array_in>
    static void img2col_3d(array_out out, array_in in) {
        BC::gpu_impl::img2col_2d<<<blocks(out.size()), threads()>>>(out, in);
    }



};


}



#endif /* GPU_CONVOLUTION_H_ */
