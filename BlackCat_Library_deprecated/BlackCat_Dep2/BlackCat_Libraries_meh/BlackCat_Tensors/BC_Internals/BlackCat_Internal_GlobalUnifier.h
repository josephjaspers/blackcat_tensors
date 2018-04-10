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

//Internal
//#include "BlackCat_Internal_Definitions.h"
//#include "BlackCat_Internal_IO.h"

//Expressions
//#include "BC_ArrayWrappers/ArrayType_Base.h"
#include "BC_Expressions/Expression_Base.h"
#include "BC_Expressions/Expression_Unary_Negation.h"
#include "BC_Expressions/Expression_Binary_Functors.h"
#include "BC_Expressions/Expression_Binary_Pointwise_Same.h"
#include "BC_Expressions/Expression_Binary_Pointwise_Scalar.h"
#include "BC_Expressions/Expression_Binary_VVmul.h"
#include "BC_Expressions/Expression_Unary_MatrixTransposition.h"

//Array Wrappers
#include "BC_Mathematics_CPU/Mathematics_CPU.h"

//Meta-templates Functions
#include "BC_MetaTemplateFunctions/Adhoc.h"
#include "BC_MetaTemplateFunctions/Complex.h"
#include "BC_MetaTemplateFunctions/Simple.h"

//Shapes (Inheritance Lv 1)
#include "BC_Shape/Static_Shape.h"
#include "BC_Shape/Static_Shape_Impl.h"
#include "BC_Shape/Shape_DefaultLD_Impl.h"

////Inheritance Lv 2 - 4
//#include "BC_Tensor_InheritanceHierachy/Tensor_Lv2_Traits.h"
//#include "BC_Tensor_InheritanceHierachy/Tensor_Lv3_Array_Utility.h"
//#include "BC_Tensor_InheritanceHierachy/Tensor_Lv3_Expr_Utility.h"
//#include "BC_Tensor_InheritanceHierachy/Tensor_Lv4_Core.h"
//
////Primary Classes (Lv 5)
//#include "BC_PrimaryClasses/BC_Tensor_Scalar.h"
//#include "BC_PrimaryClasses/BC_Tensor_Vector.h"
//#include "BC_PrimaryClasses/BC_Tensor_Matrix.h"
//#include "BC_PrimaryClasses/BC_Tensor_Cube.h"


#endif /* BLACKCAT_INTERNAL_GLOBALUNIFIER_H_ */
