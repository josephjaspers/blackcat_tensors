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


#define BC_TEST_START\
	{\
		auto test = [&]()
#define BC_TEST_END(msg)\
		;try { \
		if (!test()) {\
			std::cout << "TEST FAILURE: " << msg  << std::endl;\
			errors++; \
		} else {\
			std::cout << "TEST SUCCESS: " << msg  << std::endl;\
		}\
} catch (...) { std::cout << "TEST ERROR: " << msg  << std::endl; errors++; } }\



#define BC_TEST_BODY_HEAD std::cout << '\n' << __PRETTY_FUNCTION__ << '\n'; BC::size_t  errors = 0;
#define BC_TEST_BODY_TAIL return errors;

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
