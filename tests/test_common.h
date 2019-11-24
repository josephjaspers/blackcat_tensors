/*
 * test_common.h
 *
 *  Created on: Dec 8, 2018
 *      Author: joseph
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include <string>
#include "../include/BlackCat_Tensors.h"

namespace BC {
namespace tests {

#define BC_TEST_ON_STARTUP(...)\
		using BC_TEST_ON_STARTUP = std::true_type;\
		auto BC_TEST_ON_STARTUP = [&](){ __VA_ARGS__; }; \

#define BC_TEST_DEF(...)\
	{\
	\
		static_assert(BC_TEST_ON_STARTUP::value, \
				"BC_TEST_ON_STARTUP HAS NOT BEEN DECLARED, "\
				"use BC_TEST_ON_STARTUP(...) above this TEST_DEF")\
		numtests++;\
		BC_TEST_ON_STARTUP();\
		auto test = [&]() { __VA_ARGS__ };\
		try { \
		if (!test()) {\
			std::cout << "TEST FAILURE: " #__VA_ARGS__  << std::endl;\
			errors++; \
		} else {\
			std::cout << "test success: " #__VA_ARGS__  << std::endl;\
		}\
} catch (...) { std::cout << "TEST ERROR: " #__VA_ARGS__  << '\n'; errors++; } }


#define BC_TEST_BODY_HEAD \
	std::cout << '\n' << __PRETTY_FUNCTION__ << '\n'; \
	using BC_ASSERT_TEST_BODY_HEAD =  void;\
	int errors = 0; \
	int numtests = 0; \

#define BC_TEST_BODY_TAIL\
	std::cout << "Tests passed: " << numtests - errors << "/" << numtests << "\n";\
	static_assert(std::is_void<BC_ASSERT_TEST_BODY_HEAD>::value, \
			"BC_TEST_BODY_HEAD is not defined in function");\
	return errors;
}
}



#endif /* TEST_COMMON_H_ */
