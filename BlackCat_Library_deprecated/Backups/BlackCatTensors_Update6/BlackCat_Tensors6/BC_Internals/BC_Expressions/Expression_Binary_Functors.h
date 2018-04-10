/*
 * BC_Expression_Binary_Functors.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BINARY_FUNCTORS_H_
#define EXPRESSION_BINARY_FUNCTORS_H_

	/*
	 * Ideally these will become identity classes (no impl)
	 */
namespace BC {
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


#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */
