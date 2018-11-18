/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef GPU_UTILITY_H_
#define GPU_UTILITY_H_


namespace BC {

template<class core_lib>
struct GPU_Utility {


    static void barrier() {
        cudaDeviceSynchronize();
    }

    template<class ranks>
    static int calc_size(ranks R, int order) {
        if (order == 0) {
            return 1;
        }

        int sz = 1;
        for (int i = 0; i < order; ++i) {
            sz *= R[i];
        }
        return sz;
    }

    template<class RANKS, class os>
    static void print(const float* ary, const RANKS ranks,const os outer, int order, int print_length) {
        int sz = calc_size(ranks, order);
        float* print = new float[sz];

        cudaMemcpy(print, ary, sizeof(float) * sz, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        BC::IO::print(print, ranks, outer, order, print_length);
        delete[] print;
    }
    template<class RANKS, class     os>
    static void printSparse(const float* ary, const RANKS ranks, const os outer, int order, int print_length) {
        int sz = calc_size(ranks, order);
        float* print = new float[sz];
        cudaMemcpy(print, ary, sizeof(float) * sz, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();


        BC::IO::printSparse(print, ranks, outer, order, print_length);
        delete[] print;
    }


};


}


#endif /* GPU_UTILITY_H_ */
