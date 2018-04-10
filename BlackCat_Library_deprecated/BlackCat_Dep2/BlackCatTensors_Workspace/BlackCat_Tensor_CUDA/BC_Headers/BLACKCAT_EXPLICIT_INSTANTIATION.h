/*
 * File:   BLACKCAT_EXPLICIT_INSTANTIATION.h
 * Author: joseph
 *
 * Created on July 31, 2017, 7:54 PM
 */

#ifndef BLACKCAT_EXPLICIT_INSTANTIATION_H
#define BLACKCAT_EXPLICIT_INSTANTIATION_H

/*
 * Blackcat_Define.h
 *
 *  Created on: Aug 16, 2017
 *      Author: joseph
 */

/* DISABLE STANDARD RUNTIME CHECKS*/
//#define BLACKCAT_DISABLE_RUNTIME_CHECKS

/*DISABLE ADVANCED RUNETIME CHECKS  -- bounds checking for data accessor */
//#define BLACKCAT_DISABLE_ADVANCED_CHECKS

//#endif /* BLACKCAT_DEFINE_H_ */
#include "BLACKCAT_CPU_MATHEMATICS.h"
#include "CPU.h"
template class CPU_MATHEMATICS<double>;
template class CPU_MATHEMATICS<float>;
template class CPU_MATHEMATICS<unsigned>;
template class CPU_MATHEMATICS<signed>;


#include "Tensor.h"
template class Tensor<double, CPU>;
template class Tensor<float, CPU>;
template class Tensor<unsigned, CPU>;
template class Tensor<signed, CPU>;

#include "Scalar.h"
template class Scalar<double, CPU>;
template class Scalar<unsigned, CPU>;
template class Scalar<signed, CPU>;
template class Scalar<float, CPU>;

#include "Matrix.h"
template class Matrix<double>;
template class Matrix<unsigned>;
template class Matrix<signed>;
template class Matrix<float>;

#include "Vector.h"
template class Vector<double>;
template class Vector<unsigned>;
template class Vector<signed>;
template class Vector<float>;



#endif /* BLACKCAT_EXPLICIT_INSTANTIATION_H */

