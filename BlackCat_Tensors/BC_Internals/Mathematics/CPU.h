/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MATHEMATICS_CPU_H_
#define MATHEMATICS_CPU_H_

#include "Print.h"

#include "Math_Library_Directives.h"
#include "CPU_Implementation/CPU_Utility.h"
#include "CPU_Implementation/CPU_Misc.h"
#include "CPU_Implementation/CPU_BLAS.h"
#include "CPU_Implementation/CPU_BLAS.h"
#include "CPU_Implementation/CPU_Constants.h"
#include "CPU_Implementation/CPU_Evaluator.h"
#include "CPU_Implementation/CPU_Algorithm.h"


namespace BC {

/*
 * The core CPU library,
 *
 * defines:
 *     dimensional evaluations
 *
 */

class CPU:
        public CPU_Utility<CPU>,
        public CPU_Misc<CPU>,
        public CPU_BLAS<CPU>,
        public CPU_Constants<CPU>,
        public CPU_Evaluator<CPU>,
        public CPU_Algorithm<CPU>
{


public:

    static constexpr int SINGLE_THREAD_THRESHOLD = 8192;

};
}
#endif
 /* MATHEMATICS_CPU_H_ */
