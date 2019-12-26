/*
 * test_common.h
 *
 *  Created on: Dec 8, 2018
 *      Author: joseph
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include <string>
#include "../blackcat/tensors.h"
#include "../blackcat/shape.h"

namespace bc {
namespace tests {

static std::string sz80_substring(std::string test) {
	return test.size() < 80 - 3 ?
			test + "..." :
			test.substr(0, 80-3) + "...";
}

static std::string sz60_test_buffer() {
	return std::string(60,'-');
}

#define BC_TEST_BODY_HEAD                                                 \
	std::cout << "\n\n" << __PRETTY_FUNCTION__ << '\n';                   \
	bc::print(sz60_test_buffer());                                        \
	using BC_ASSERT_TEST_BODY_HEAD =  std::true_type;                     \
	int bc_test_num_errors = 0;                                           \
	int bc_test_index = 0;

#define BC_TEST_ON_STARTUP                                                \
		using BC_TEST_ASSERT_ON_STARTUP = std::true_type;                 \
		auto BC_TEST_STARTER = [&]()

#define BC_TEST_DEF(...)                                                  \
	{                                                                     \
                                                                          \
		static_assert(BC_TEST_ASSERT_ON_STARTUP::value,                   \
				"BC_TEST_ON_STARTUP HAS NOT BEEN DECLARED, "              \
				"use BC_TEST_ON_STARTUP(...) above this TEST_DEF");       \
		bc_test_index++;                                                  \
		BC_TEST_STARTER();                                                \
		auto test = [&]() { __VA_ARGS__ };                                \
		try {                                                             \
			if (!test()) {                                                \
				bc::print(sz80_substring("TEST FAILURE: "#__VA_ARGS__));  \
				bc_test_num_errors++;                                     \
			} else {                                                      \
				bc::print(sz80_substring("test success: "#__VA_ARGS__));  \
			}                                                             \
		} catch (...) {                                                   \
			bc::print(sz80_substring("TEST ERROR: "#__VA_ARGS__));        \
			bc_test_num_errors++;                                         \
		}                                                                 \
	}

#define BC_TEST_BODY_TAIL\
		static_assert(BC_ASSERT_TEST_BODY_HEAD::value,                    \
				"BC_TEST_BODY_HEAD is not defined in function");          \
                                                                          \
	bc::print(sz60_test_buffer());                                        \
	bc::print("Tests passed:",                                            \
			bc_test_index-  bc_test_num_errors, "/", bc_test_index);      \
	return bc_test_num_errors;
}
}



#endif /* TEST_COMMON_H_ */
