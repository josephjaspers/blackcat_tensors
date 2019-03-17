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

#define BC_LOG_VARIABLE(variable) ON_ERROR_OUTPUT_VECTOR.push_back(#variable + std::string(": ") + std::to_string(variable))

#define BC_ON_TEST_FAILURE\
	for (std::string& str : ON_ERROR_OUTPUT_VECTOR) {\
		std::cout << '\n\t' <<  str;\
	if (!ON_ERROR_OUTPUT_VECTOR.empty())\
	std::cout << '\n';\
}

#define BC_TEST_DEF(...)\
	{\
	\
		std::vector<std::string> ON_ERROR_OUTPUT_VECTOR;\
		numtests++;\
		auto test = [&]() { __VA_ARGS__ };\
		try { \
		if (!test()) {\
			std::cout << "TEST FAILURE: " #__VA_ARGS__  << std::endl;\
			errors++; \
		} else {\
			std::cout << "TEST SUCCESS: " #__VA_ARGS__  << std::endl;\
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

#define BC_ON_ERROR(argument)

template<class arg>
void print(const arg& arg_) {
	std::cout << arg_ << "\n";
}

template<class arg, class... args>
void print(const arg& a, const args&... arg_) {
	std::cout << a << " ";
	print(arg_...);
}

}
}



#endif /* TEST_COMMON_H_ */
