/*
 * BC_Expression_Binary_Functors.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_FUNCTORS_H_
#define EXPRESSION_BINARY_FUNCTORS_H_

#include "BlackCat_Internal_Definitions.h"
#include <iostream>

namespace BC {
template<class T> struct rm_const { using type = T; };
template<class T> struct rm_const<const T&> { using type = T&; };

/*
 * 0 = a Tensor (base case)
 * 1 = function
 * 2 = Multiplication/division
 * 3 = addition/subtraction
 * 4 = assignments
 *
 */

	struct scalar_mul {
		//this is just a flag for dotproduct, it is the same as multiplication though
		template<class lv, class rv> __BCinline__ auto operator ()(lv l, rv r) const {
			return l * r;
		}
	};


	struct add {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l + r;
		}
	};

	struct mul {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l * r;
		}
	};

	struct sub {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l - r;
		}
	};

	struct div {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l / r;
		}
	};
	struct assign {
		template<class lv, class rv> __BCinline__ auto& operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) = r);
		}
	};

	struct combine {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l;
		}
	};
	struct add_assign {
		static constexpr int PRIORITY() { return 4; }
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return l += r;
		}
	};

	struct mul_assign {
		static constexpr int PRIORITY() { return 4; }
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return l *= r;
		}
	};

	struct sub_assign {
		static constexpr int PRIORITY() { return 4; }
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return l -= r;
		}
	};

	struct div_assign {
		static constexpr int PRIORITY() { return 4; }
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return l /= r;
		}
	};
}


#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

