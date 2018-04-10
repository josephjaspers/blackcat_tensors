/*
 * BC_MetaTemplate_Simple.h
 *
 *  Created on: Dec 12, 2017
 *      Author: joseph
 */

#ifndef SIMPLE_H_
#define SIMPLE_H_
namespace BC {
namespace BC_MTF {

	template<int a, int b> struct max { static constexpr int value = a > b ? a : b; };
	template<int a, int b> struct min { static constexpr int value = a < b ? a : b;  };

	template<int ... dims> 		  struct first;
	template<int f, int ... dims> struct first<f, dims...> { static constexpr int value = f; };

	template<bool con> struct isFalse { static constexpr bool conditional = !con;  };
	template<bool con> struct isTRUE  { static constexpr bool conditional = con;  };

	template<bool, bool> struct OR 				 { static constexpr bool conditional = true ;};
	template<>			 struct OR<false, false> { static constexpr bool conditional = false;};

	template<int a, int b> struct equal 		 {	static constexpr bool conditional = a == b; };
	template<int a, int b> struct greater_than   {	static constexpr int  value = a > b ? a : b;
												 	static constexpr bool conditional = a > b ? true : false; };
	template<int a, int b> struct less_than      { 	static constexpr int  value = a < b ? a : b;
													static constexpr bool conditional = a < b ? true : false; };

	template<class>struct front;

	template<template<class...> class list, class... set, class first>
	struct front<list<first, set...>>
	{ using type = first; };

	template<class> struct isPrimitive { static constexpr bool conditional = false; };
	template<> struct isPrimitive<bool> 				{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<short> 				{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<unsigned short> 		{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<int> 					{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<unsigned> 			{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<long> 				{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<unsigned long> 		{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<long long> 			{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<unsigned long long> 	{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<char> 				{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<unsigned char>		{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<float> 				{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<double> 				{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<long double> 			{ static constexpr bool conditional = true; };
	template<> struct isPrimitive<wchar_t> 				{ static constexpr bool conditional = true; };

	template<class T> struct remove_ref 	 { using type = T; };
	template<class T> struct remove_ref<T&>  { using type = T; };
	template<class T> struct remove_ref<T&&> { using type = T; };
	template<typename _this> struct ID { using type = typename remove_ref<_this>::type; };		///I like this ends up reading BC_MTF::ID<this>::type


	template<class>
	class List_Traits {
		static constexpr bool isList = false;
		static constexpr bool isIntList = false;
		static constexpr bool isClassList = false;
	};
	template< template<int...> class list, int... set>
	struct List_Traits<list<set...>>{
		static constexpr bool isList = true;
		static constexpr bool isIntList = true;
		static constexpr bool isClassList = false;
	};
	template< template<class...> class list, class... set>
	struct List_Traits<list<set...>>{
		static constexpr bool isList = true;
		static constexpr bool isIntList = false;
		static constexpr bool isClassList = true;
	};


}
}
#endif /* SIMPLE_H_ */
