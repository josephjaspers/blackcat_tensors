/*
 * BC_Internal_Include.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BLACKCAT_INTERNAL_GLOBALUNIFIER_H_
#define BLACKCAT_INTERNAL_GLOBALUNIFIER_H_
/*
 * DO NOT REORDER:
 * COMPILATION IS DEPENDANT ON INCLUSION ORDER
 *
 * 																						bug - Eclipses rearranges includes when refactoring
 * ORDER FOR CORRECT COMPILATION (if these includes are out of order rearrange)
 *
 *  1) BlackCat_Definitions
 *  2) ArrayWrappers
 *  3) Expressions
 *  4) Mathematics
 *  5) Meta-template functions
 *  6) Shape
 *  7) Inheritance Type_helpers
 *  8) Inheritance Hierarchy
 *  9) Primary Classes
 *
 *
 */

#define  BLACKCAT_GPU_ENABLED

#include "BlackCat_Internal_Definitions.h"
#include "BlackCat_ForwardDecls.h"

//#define BLACKCAT_CPU_ONLY
#include "BC_MathLibraries/Mathematics_CPU.h"
#include "BC_MathLibraries/Mathematics_GPU.cu"
#include "BC_MathLibraries/Mathematics_mGPU.cu"


#include "BlackCat_ForwardDecls.h"



//Expressions
#include "BC_MathLibraries/Mathematics_CPU.h"
#include "BC_MathLibraries/Mathematics_GPU.cu"


#include "BC_Expressions/Expression_Base.h"
#include "BC_Expressions/Expression_Unary_Negation.h"
#include "BC_Expressions/Expression_Binary_Functors.cu"
#include "BC_Expressions/Expression_Binary_Pointwise_Same.cu"
#include "BC_Expressions/Expression_Binary_Pointwise_Scalar.cu"
#include "BC_Expressions/Expression_Binary_Dotproduct.cu"

#include "BC_Expressions/Expression_Binary_VVmul.h"
#include "BC_Expressions/Expression_Unary_MatrixTransposition.cu"

//Array Wrappers

//Meta-templates Functions
#include "BC_MetaTemplateFunctions/Simple.h"
#include "BC_MetaTemplateFunctions/Complex.h"
#include "BC_MetaTemplateFunctions/Adhoc.h"

//Inlucde the mathlibs

//BC_core
#include "BC_Core/Scalar.h"
#include "BC_Core/Vector.h"
#include "BC_Core/Matrix.h"







#endif /* BLACKCAT_INTERNAL_GLOBALUNIFIER_H_ */
