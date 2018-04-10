/*
 * Dynamic_List.h
 *
 *  Created on: Dec 28, 2017
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_SHAPE_DYNAMICINTLIST_H_
#define BC_INTERNALS_BC_SHAPE_DYNAMICINTLIST_H_


namespace BC {

	template<class...>
	struct first;

	template<class f, class... set>
	struct first<f, set...> {
		using type = f;
	};

	//Head and Body
	template<class first, class... IntList>
	struct DynamicIntList{
		int data = first;
		DynamicIntList<IntList...> next;

		int& operator [] (int index) {
			return (&data)[index];
		}
		const int& operator [] (int index) const{
			return (&data)[index];
		}
	};
	//Tail
	template<class first>
	struct DynamicIntList<first> {
		int data = first;

		int& operator [] (int index) {
			return (&data)[index];
		}
		const int& operator [] (int index) const{
			return (&data)[index];
		}

	};

	//Generator
	template<class... set>
	auto GenerateDynamicIntList(set... integers) {
		return DynamicIntList<set...>();
	}





}


#endif /* BC_INTERNALS_BC_SHAPE_DYNAMICINTLIST_H_ */
