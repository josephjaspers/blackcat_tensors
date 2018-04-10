#include <string>
#include <vector>
#include <iostream>
#include "../Structures/list_comprehension.h"
#include "../Structures/thread_map.h"
#include <pthread.h>
template<class T>
using List = std::vector<T>;
using Str = std::string;
using BC::Structures::lc;
int main() {


	List<Str> list = { "12", "13", "23", "44",  };

	auto l2 = lc(
			list,									//source
			[](auto x) { return x.substr(1,2); },	//modifier
			[](auto x) { return x[0] == '1'; }		//conditional
	);


	for (int i = 0; i < l2.size(); ++i) {
		std::cout << l2[i] << std::endl;
	}



	std::cout << pthread_self() << std::endl;

}
