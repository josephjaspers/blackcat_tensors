/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef __CUDACC__
#ifndef MATHEMATICS_GPU_H_
#define MATHEMATICS_GPU_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Print.h"
#include "GPU_Implementation/GPU_impl.cu"
#include "GPU_Implementation/GPU_BLAS.h"
#include "GPU_Implementation/GPU_Misc.h"
#include "GPU_Implementation/CUDA_Allocator.h"
#include "GPU_Implementation/GPU_Constants.h"
#include "GPU_Implementation/GPU_Evaluator.h"

namespace BC {

class GPU :
	public GPU_Misc<GPU>,
	public CUDA_Allocator,
	public GPU_BLAS<GPU>,
	public GPU_Constants<GPU>,
	public GPU_Evaluator<GPU>{
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

#endif /* MATHEMATICS_CPU_H_ */

#endif //if cudda cc defined
