/*
 * BC_Core.h
 *
 *  Created on: Feb 14, 2018
 *      Author: joseph
 */

#ifndef BC_CORE_H_
#define BC_CORE_H_
#include <type_traits>

namespace BC {




	template<class derived, class T, class ml>
	class TensorBase  {

		static constexpr bool rank = derived::getRank();

		operator const derived&() { return static_cast<derived&>(*this); }
		operator 	   derived&() { return static_cast<derived&>(*this); }

		struct as {
			template<class a, class b, class c,
					class A, class B, class C,

			template<class,class,class> class tens>
			tensor<A, B, C> getShell(class A, class B, class C, tens<a,b,c>);

			template<class A, class B>
			using type = std::declval(getShell(std::declval(A), std::declval(B), std::declval(TensorBase)));
		};


			  derived& deriv() 		 { return *this; }
		const derived& deriv() const { return *this; }


		struct DynamicShape {

		};

	};

}
foo

#endif /* BC_CORE_H_ */
