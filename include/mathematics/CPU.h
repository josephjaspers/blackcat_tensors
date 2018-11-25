/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MATHEMATICS_CPU_H_
#define MATHEMATICS_CPU_H_

#include "Math_Library_Directives.h"
#include "cpu_implementation/CPU_Misc.h"
#include "cpu_implementation/CPU_BLAS.h"
#include "cpu_implementation/CPU_BLAS.h"
#include "cpu_implementation/CPU_Constants.h"
#include "cpu_implementation/CPU_Evaluator.h"


namespace BC {

/*
 * The core CPU library,
 *
 * defines:
 *     dimensional evaluations
 *
 */

class CPU:
        public CPU_Misc<CPU>,
        public CPU_BLAS<CPU>,
        public CPU_Constants<CPU>,
        public CPU_Evaluator<CPU>
{


public:

    static constexpr int SINGLE_THREAD_THRESHOLD = 8192;

	template<class T, class U, class V>
	static void copy(T* to, U* from, V size) {
		__BC_omp_for__
		for (int i = 0; i < size; ++i) {
			to[i] = from[i];
		}

		__BC_omp_bar__
	}

};
}
#endif
 /* MATHEMATICS_CPU_H_ */
