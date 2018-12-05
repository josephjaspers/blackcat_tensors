/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MATHEMATICS_CPU_H_
#define MATHEMATICS_CPU_H_

#include "host_impl/BLAS.h"
#include "host_impl/Constants.h"
#include "host_impl/Evaluator.h"


namespace BC {
namespace evaluator {

/*
 * The core CPU library,
 *
 * defines:
 *     dimensional evaluations
 *
 */

class Host:
        public host_impl::BLAS<Host>,
        public host_impl::Constants<Host>,
        public host_impl::Evaluator<Host>
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
}
#endif
 /* MATHEMATICS_CPU_H_ */
