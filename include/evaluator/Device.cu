/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef __CUDACC__
#ifndef BC_MATHEMATICS_DEVICE_H_
#define BC_MATHEMATICS_DEVICE_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "device_impl/Impl.cu"
#include "device_impl/BLAS.h"
#include "device_impl/Misc.h"
#include "device_impl/Constants.h"
#include "device_impl/Evaluator.h"

namespace BC {
namespace evaluator {

class Device :
    public device_impl::Misc<Device>,
    public device_impl::BLAS<Device>,
    public device_impl::Constants<Device>,
    public device_impl::Evaluator<Device> {
public:

    static constexpr int CUDA_BASE_THREADS = 256;

    static int blocks(int size) {
        return 1 + (int)(size / CUDA_BASE_THREADS);
    }
    static int threads(int sz = CUDA_BASE_THREADS) {
        return sz > CUDA_BASE_THREADS ? CUDA_BASE_THREADS : sz;
    }

};

}
}

#endif /* MATHEMATICS_CPU_H_ */

#endif //if cudda cc defined
