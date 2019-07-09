/*
 * Common.h
 *
 *  Created on: Jun 30, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TYPETRAITS_COMMON_H_
#define BLACKCAT_TYPETRAITS_COMMON_H_

#include <type_traits>
#include <tuple>

namespace BC {
namespace traits {
namespace common {

	using std::enable_if_t;
	using std::true_type;
	using std::false_type;
	using std::is_same;
	using std::declval;
	using std::is_const;
	using std::conditional_t;
	using std::tuple;
	using std::forward;

}
}
}


#endif /* COMMON_H_ */
