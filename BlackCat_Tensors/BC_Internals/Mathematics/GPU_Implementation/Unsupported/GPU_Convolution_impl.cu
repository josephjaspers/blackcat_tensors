/*
 * Image2Col.h
 *
 *  Created on: Jul 5, 2018
 *      Author: joseph
 */

#ifndef IMAGE2COL_H_
#define IMAGE2COL_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace BC {
namespace gpu_impl {


//2 dimensional image2col accepts a matrix (in) as the image and a 4d matrix out reshaped for the output matrix(rows, cols, row_positions, col_positions)
template<class img_out, class img_in>
void img2col_2d(img_out out, img_in in) {
    static_assert(img_out::DIMS() == 4 && img_in::DIMS() == 2, "img2col 2d requires a 4d and 2d tensors");
    //number of column positions
    int cp = blockIdx.z * blockDim.z + threadIdx.z;
    for (; cp < out.dimension(3); cp += blockDim.z * gridDim.z) {

        //number of row positions
        int rp = blockIdx.y * blockDim.y + threadIdx.y;
        for (; rp < out.dimension(2); rp += blockDim.y * gridDim.y) {

            //number of kernel cols
            int c = blockIdx.x * blockDim.x + threadIdx.x;
            for (; c < out.dimesnion(1); c += blockDim.x * gridDim.x) {

                //number of kernel rows
                for (int r = 0; r < out.dimension(0); ++r) {
                    out(r, c, rp, cp) = in(r + rp, c + cp);
                }
            }

        }
    }
}
template<class img_out, class img_in>
void img2col_3d(img_out out, img_in in) {
    static_assert(img_out::DIMS() == 5 && img_in::DIMS() == 3, "img2col 2d requires a 4d and 2d tensors");
    //number of depth positions
    int dp = blockIdx.z * blockDim.z + threadIdx.z;
    for (; dp < out.dimension(3); dp += blockDim.z * gridDim.z) {
        //number of columns positions
        int cp = blockIdx.y * blockDim.y + threadIdx.y;
        for (; cp < out.dimension(2); cp += blockDim.y * gridDim.y) {
            //number of row positions
            int rp = blockIdx.x * blockDim.x + threadIdx.x;
            for (; rp < out.dimesnion(1); rp += blockDim.x * gridDim.x) {
                for (int d = 0; d < out.dimension(0); ++d)
                    for (int c = 0; c < out.dimension(0); ++c)
                        for (int r = 0; r < out.dimension(0); ++r)
                            out(r,c,d,rp,cp,dp) = in(r + rp, c + cp, d + dp);
            }

        }
    }
}

}
}

#endif /* IMAGE2COL_H_ */
