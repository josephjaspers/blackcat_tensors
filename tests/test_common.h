/*
 * test_common.h
 *
 *  Created on: Dec 8, 2018
 *      Author: joseph
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include "../include/BlackCat_Tensors.h"

namespace BC {
namespace tests {

#define BC_TEST_DEF(code)\
	{\
		auto test = [&]() { code };\
		try { \
		if (!test()) {\
			std::cout << "TEST FAILURE: " #code  << std::endl;\
			errors++; \
		} else {\
			std::cout << "TEST SUCCESS: " #code  << std::endl;\
		}\
} catch (...) { std::cout << "TEST ERROR: " #code  << std::endl; errors++; } }\

}
}



#endif /* TEST_COMMON_H_ */
