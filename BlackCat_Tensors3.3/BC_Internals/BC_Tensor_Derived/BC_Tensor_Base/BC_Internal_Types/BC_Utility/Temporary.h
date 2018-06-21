/*
 * Temporary_Array.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef TEMPORARY_ARRAY_H_
#define TEMPORARY_ARRAY_H_

namespace BC {
namespace internal {

/*
 * used as a wrapper for BC_Arrays, enalbes the expression tree to traverse through and delete the correct
 */

template<class data_t>
struct temporary : data_t {
	using data_t::data_t;
};

}
}



#endif /* TEMPORARY_ARRAY_H_ */
