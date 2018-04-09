#include <string>
#include <vector>
#include <iostream>
#include "ListComprehension.h"

template<class T>
using List = std::vector<T>;
using Str = std::string;
using BC::LC::lc;
int main() {


	List<Str> list = { "ald", "Tts", "ams",  };

	auto l2 = lc(
			list,									//source
			[](auto x) { return x.substr(1,3); },	//modifier
			[](auto x) { return x[0] == 'a'; }		//conditional
	);


	for (int i = 0; i < l2.size(); ++i) {
		std::cout << l2[i] << std::endl;
	}



}
