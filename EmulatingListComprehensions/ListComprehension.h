#ifndef LIST_COMPREHENSION
#define LIST_COMPREHENSION

#include <vector>

namespace BC {
namespace LC {

//take a parameter always return true
struct default_conditional {
	template<class T>
	auto operator () (const T& asd) { return true; }
};
static constexpr default_conditional dc;

template<class L, class F, class C = default_conditional>
static auto lc(std::vector<L> list, F lamda, C conditional = default_conditional()) {


	std::vector<decltype(lamda(list[0]))> new_list(0);

		for (std::size_t i = 0; i < list.size(); ++i) {
			if (conditional(list[i]))
				new_list.push_back(lamda(list[i]));
		}
		return new_list;
	};
}
}


#endif
