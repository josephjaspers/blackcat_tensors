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

struct basic_operation {};
struct assignment_operation : basic_operation{};
struct linear_operation : basic_operation {};
struct linear_assignment_operation : linear_operation, assignment_operation {};
struct BLAS_Function {
	///BLAS functions cannot be lazy evaluated
	using requires_greedy_evaluation = std::true_type;
};

}
}



#endif /* TAGS_H_ */
