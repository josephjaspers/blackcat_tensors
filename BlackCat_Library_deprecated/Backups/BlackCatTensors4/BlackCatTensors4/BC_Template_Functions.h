/*
 * BC_Template_Functions.h
 *
 *  Created on: Nov 27, 2017
 *      Author: joseph
 */

#ifndef BC_TEMPLATE_FUNCTIONS_H_
#define BC_TEMPLATE_FUNCTIONS_H_

namespace BC {

	template<class T, bool conditional>
	struct enable_if {
		//false
	};
	template<class T>
	struct enable_if<T, true> {
		using type = T;
	};

	template<class T>
	struct isPrimitive {
		static constexpr bool value = false;
	};
	template<>
	struct isPrimitive<int> {
		static constexpr bool value = true;
	};
	template<>
	struct isPrimitive<unsigned> {
		static constexpr bool value = true;
	};
	template<>
	struct isPrimitive<float> {
		static constexpr bool value = true;
	};
	template<>
	struct isPrimitive<long> {
		static constexpr bool value = true;
	};
	template<>
	struct isPrimitive<long long> {
		static constexpr bool value = true;
	};
	template<>
	struct isPrimitive<double> {
		static constexpr bool value = true;
	};

}

#endif /* BC_TEMPLATE_FUNCTIONS_H_ */
