/*
 * BLAS_Priorities.h
 *
 *  Created on: Jun 12, 2018
 *      Author: joseph
 */

#ifndef BLAS_PRIORITIES_H_
#define BLAS_PRIORITIES_H_

namespace BC{

//priority -1 == non-rotateable (injection not legal)
//priority 0 == low
//priority 1 == high
/*
 * a low priority may be after a high priority (ensure priority order)
 * but a low priority preceding a high-priority operation will result in non-injectable segment
 *
 * This module assigns arithmetic values to operands to handle apropriate precedence in injections
 */


template<class> struct PRIORITY { static constexpr int value = -1;};
template<> struct PRIORITY<oper::add> { static constexpr int value = 0;
static constexpr int alpha_modifier = 1;
static constexpr int beta_modifier = 1;};
template<> struct PRIORITY<oper::sub> { static constexpr int value = 0;
static constexpr int alpha_modifier = -1;
static constexpr int beta_modifier = 1;};
template<> struct PRIORITY<oper::mul> { static constexpr int value = 1; };
template<> struct PRIORITY<oper::div> { static constexpr int value = 1; };
template<> struct PRIORITY<oper::add_assign> { static constexpr int value = 0;};
template<> struct PRIORITY<oper::sub_assign> { static constexpr int value = 0;}; //this will eventually be allowed but for now NO
template<> struct PRIORITY<oper::assign> { static constexpr int value = 1; };


template<class T>
static constexpr int alpha_modifier() {
	return PRIORITY<std::decay_t<T>>::alpha_modifier;
}
template<class T>
static constexpr int beta_modifier() {
	return PRIORITY<std::decay_t<T>>::beta_modifier;
}



}



#endif /* BLAS_PRIORITIES_H_ */
