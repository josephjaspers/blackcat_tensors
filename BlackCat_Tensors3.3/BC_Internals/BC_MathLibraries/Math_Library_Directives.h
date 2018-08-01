/*
 * Math_Library_Directives.h
 *
 *  Created on: Aug 1, 2018
 *      Author: joseph
 */

#ifndef MATH_LIBRARY_DIRECTIVES_H_
#define MATH_LIBRARY_DIRECTIVES_H_


#ifdef _OPENMP
#define __BC_omp_for__ _Pragma("omp parallel for")
#define __BC_omp_bar__ _Pragma("omp barrier")
#else
#define __BC_omp_for__
#define __BC_omp_bar__
#endif

#endif /* MATH_LIBRARY_DIRECTIVES_H_ */
