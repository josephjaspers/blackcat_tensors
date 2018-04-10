/*
 * BC_Expression_Functions.h
 *
 *  Created on: Nov 21, 2017
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_FUNCTORS_H_
#define BC_EXPRESSION_FUNCTORS_H_

#include "BC_Tensor_Super_King.h"
namespace BC {

	//Base structs ---- we don't actually use any of their implementations
	struct add {
		template<class lv, class rv>
		inline __attribute__((always_inline)) auto operator ()(lv l, rv r) const {
			return l + r;
		}

		template<class eval_to, class lv, class rv>
		inline __attribute__((always_inline)) void operator ()(eval_to& to, lv l, rv r) const {
			to = l + r;
		}
	};

	struct mul {
		template<class lv, class rv>
		inline __attribute__((always_inline)) auto operator ()(lv l, rv r) const {
			return l * r;
		}
		template<class eval_to, class lv, class rv>
		inline __attribute__((always_inline)) void operator ()(eval_to& to, lv l, rv r) const {
			to = l * r;
		}
	};

	struct sub {
		template<class lv, class rv>
		inline __attribute__((always_inline)) auto operator ()(lv l, rv r) const {
			return l - r;
		}
		template<class eval_to, class lv, class rv>
		inline __attribute__((always_inline)) void operator ()(eval_to& to, lv l, rv r) const {
			to = l - r;
		}
	};

	struct div {
		template<class lv, class rv>
		inline __attribute__((always_inline)) auto operator ()(lv l, rv r) const {
			return l / r;
		}
		template<class eval_to, class lv, class rv>
		inline __attribute__((always_inline)) void operator ()(eval_to& to, lv l, rv r) const {
			to = l / r;
		}
	};
}

#endif /* BC_EXPRESSION_FUNCTORS_H_ */
