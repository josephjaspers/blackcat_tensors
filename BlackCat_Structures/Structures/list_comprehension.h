#ifndef LIST_COMPREHENSION
#define LIST_COMPREHENSION

#include <vector>

namespace BC {
namespace Structures {

//take a parameter always return true
struct default_conditional {
	template<class T>
	auto operator () (const T& asd) { return true; }
};

template<class...> struct shell;

template<template<class...> class shell_, class head, class... tail>
struct shell<shell_<head, tail...>> {

	template<class T>
	using type = shell<T, tail...>;

};


template<class T, class F, class C = default_conditional>
static auto lc(std::vector<T>& list, F lamda, C conditional = default_conditional()) {

	std::vector<decltype(lamda(list[0]))> new_list;

		for (std::size_t i = 0; i < list.size(); ++i) {
			if (conditional(list[i]))
				new_list.push_back(lamda(list[i]));
		}
		return new_list;
	}

template<class T, class F, class C = default_conditional>
static auto list_comprehension(std::vector<T>& list, F lamda, C conditional = default_conditional()) {
	return lc(list, lamda, conditional);
}
}
}
#endif
