#include "ListComprehension.h"
#include <string>
#include <iostream>

/*
 * Example of c++ list comprehension
 * (Implementation in ListComprehension.h)
 */



using namespace BC::LC;



int main() {

	std::vector<std::string> list = { "dogs", "cats", "logs", "ba", "asa", "asdf" };

	auto new_list = lc(
			list, 									//source
			[](auto x) { return "append " + x; }, 	//modifier
			[](auto x) { return x[1] == 'a' ;}		//conditional
	);


	for (unsigned i = 0; i < new_list.size(); ++i) {
		std::cout << new_list[i] << std::endl;
	}
}
