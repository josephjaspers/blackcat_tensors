/*
 * Tags.h
 *
 *  Created on: Feb 17, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_OPERATIONS_TAGS_H_
#define BC_CORE_OPERATIONS_TAGS_H_



namespace BC {
namespace oper {

struct alpha_modifier_base {};
struct beta_modifier_base {};

template<int x> struct alpha_modifier : alpha_modifier_base { static constexpr int alpha_mod = x; };
template<int x> struct beta_modifier  : beta_modifier_base  { static constexpr int beta_mod  = x; };

struct assignment_operation {};
struct linear_operation {};
struct linear_assignment_operation : linear_operation, assignment_operation {};
struct BLAS_Function {};


}
}



#endif /* TAGS_H_ */
