/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MATH_LIBRARY_DIRECTIVES_H_
#define MATH_LIBRARY_DIRECTIVES_H_


#ifdef _OPENMP
	#ifndef BC_NO_OPENMP
		#define __BC_omp_for__ _Pragma("omp parallel for")
		#define __BC_omp_bar__ _Pragma("omp barrier")
	#endif
#else
	#define __BC_omp_for__
	#define __BC_omp_bar__
#endif

#endif /* MATH_LIBRARY_DIRECTIVES_H_ */
