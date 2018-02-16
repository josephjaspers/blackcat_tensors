/*
 * BC_MetaTemplate_UtilityMethods.h
 *
 *  Created on: Dec 4, 2017
 *      Author: joseph
 */

#ifndef COMPLEX_H_
#define COMPLEX_H_

/*
 * The namespaces are just for syntactic sugar of grouping each cluster of tempalte sub functions
 *
 */
namespace BC {
namespace MTF {

	namespace parameter_pack_extractor {

		template<class, class>
		struct extract_to;

		template<
			template<class...> class List1,
			template<class...> class List2,
			class...	elements1,
			class... 	elements2>
		struct extract_to<List1<elements1...>, List2<elements2...>> {
			using type = List1<elements1..., elements2...>;
		};
		template<
			template<int...> class List1,
			template<int...> class List2,
			int...	elements1,
			int... 	elements2>
		struct extract_to<List1<elements1...>, List2<elements2...>> {
			using type = List1<elements1..., elements2...>;
		};

	}

	namespace contains_ {

		template<class, class ...>
		struct contains;

		template<class T, class front, class ... list>
		struct contains<T, front, list...> {
			static constexpr bool conditional = contains<T, list...>::conditional;
		};
		template<class T, class ... list>
		struct contains<T, T, list...> {
			static constexpr bool conditional = true;
		};
		template<class T>
		struct contains<T> {
			static constexpr bool conditional = false;
		};
	}
	namespace type_of_type_ {
		template<class T, class voider>
		struct type_of_type;

		template<class T>
		struct type_of_type<T, typename T::type> {
			using type = typename T::type;
		};
	}

	namespace is_same_ {

		template<class,class>
		struct is_same {
			static constexpr bool conditional = false;
		};

		template<class T>
		struct is_same<T,T> {
			static constexpr bool conditional = true;
		};
	}
	namespace deriver_ {
		template<class ... bases>
		struct deriver;

		template<class front, class ... bases>
		struct deriver<front, bases...> : public virtual front, public virtual deriver<bases...> {};

		template<>
		struct deriver<> {};
	}

	namespace remove_from {

		/*
		 * The remove_from function recursively removes all instances of a type from a given class of <class... types>.
		 * The field type = the original  template class<class... types> class list with all instances of type T
		 * removed from parameter_pack types...
		 *
		 *
		 */

		namespace base_class {
			template<class, class >
			struct remove_from;

			template<class search, class list1, class list2>
			struct sifter;

			//if we can't find it shift over the front of the list
			template<class search, template<class ...> class list1, template<class...> class list2, class list2_front, class... elements1, class...elements2>
			struct sifter<search, list1<elements1...>, list2<list2_front, elements2...>> {
				using type = typename sifter<search, list1<elements1...,list2_front>, list2<elements2...>>::type;						//type of the list after removing all
				static constexpr bool conditional = sifter<search, list1<elements1...,list2_front>, list2<elements2...>>::conditional;//boolean if any removed
				static constexpr int value = 0 + sifter<search, list1<elements1...,list2_front>, list2<elements2...>>::value;//number of removed

			};
			//if we can find it continue removing
			template<class search, template<class ...> class list1, template<class...> class list2, class... elements1, class...elements2>
			struct sifter<search, list1<elements1...>, list2<search, elements2...>> {
				using type = typename sifter<search, list1<elements1...>, list2<elements2...>>::type;

				static constexpr int value = 1 + sifter<search, list1<elements1...>, list2<elements2...>>::value;			//number of removed
				static constexpr bool conditional = true;
			};
			//at end of list - establish the removed list as type
			template<class search, template<class ...> class list1, template<class...> class list2, class... elements1>
			struct sifter<search, list1<elements1...>, list2<>>{

				using type = list1<elements1...>;
				static constexpr int value = 0;			//number of removed
				static constexpr bool conditional = false;
			};

			template<class remove, template<class ...>class list, class... elements>
			struct remove_from<remove, list<elements...>> {

			static constexpr bool conditional = sifter<remove, list<>, list<elements...>>::conditional;
			using type = typename sifter<remove, list<>, list<elements...>>::type;
		};
	}
		namespace template_class {
			/*
			 * Very verbose, very heady.
			 */

			template<template<class...> class, class>
			struct remove_from;

			template<template<class...> class, class, class>
			struct sifter;

			//neither found nor empty
			template<template<class...> class search,
				template<class...> class shifted_list, class...shifted_elements,
			    template<class...> class         list, class...        elements,
				class focused_element>

			struct sifter<search, shifted_list<shifted_elements...>, list<focused_element, elements...>> {

				using removed_type 	= typename	sifter<search, shifted_list<shifted_elements..., focused_element>, list<elements...>>::removed_type;
				using type 			= typename 	sifter<search, shifted_list<shifted_elements..., focused_element>, list<elements...>>::type;


				static constexpr bool conditional = sifter<search, shifted_list<shifted_elements..., focused_element>, list<elements...>>::conditional;
				static constexpr int value        = sifter<search, shifted_list<shifted_elements..., focused_element>, list<elements...>>::value;
			};

			//Found
			template<template<class...> class search,
				template<class...> class shifted_list, class...shifted_elements,
			    template<class...> class         list, class...        elements,
				class... params>

			struct sifter<search, shifted_list<shifted_elements...>, list<search<params...>, elements...>> {

				using removed_type = search<params...>;
				using type = typename 				sifter<search, shifted_list<shifted_elements...>, list<elements...>>::type;

				static constexpr bool conditional = true;

				static constexpr int value = 1 +    sifter<search, shifted_list<shifted_elements...>,  list<elements...>>::value;
			};


			//empty
			template<template<class...> class search,
				template<class...> class shifted_list, class...shifted_elements,
			    template<class...> class         list>

			struct sifter<search, shifted_list<shifted_elements...>, list<>> {

				using removed_type = void;
				static constexpr bool conditional = false;
				using type = shifted_list<shifted_elements...>;

				static constexpr int value = 0;
			};
		}

}

using namespace contains_;
using namespace type_of_type_;
using namespace parameter_pack_extractor;
using namespace is_same_;
using namespace deriver_;
}
}
#endif /* COMPLEX_H_ */
