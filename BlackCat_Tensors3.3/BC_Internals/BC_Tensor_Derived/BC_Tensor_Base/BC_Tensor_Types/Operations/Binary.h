/*
 * BC_Expression_Binary_Functors.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_BINARY_FUNCTORS_H_
#define EXPRESSION_BINARY_FUNCTORS_H_

namespace BC {

template<class T, class U, class... Ts>
struct  inherits_from {
	static constexpr bool conditional = std::is_base_of<U, T>::value && inherits_from<T, Ts...>::conditional;

};
template<class T, class U>
struct  inherits_from<T, U> {
	static constexpr bool conditional = std::is_base_of<U, T>::value;
};

template<class T, class... Ts>
using enable_if_is = std::enable_if_t<inherits_from<T, Ts...>::conditional>;



namespace oper {
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
class assignment {};
class matrix_oper {};

	struct scalar_mul : matrix_oper {
		//this is just a flag for dotproduct, it is the same as multiplication though
		template<class lv, class rv> __BCinline__ auto operator ()(lv l, rv r) const {
			return l * r;
		}
	};


	struct add : matrix_oper {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l + r;
		}
	};

	struct mul {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l * r;
		}
	};

	struct sub : matrix_oper {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l - r;
		}
	};

	struct div {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l / r;
		}
	};
	struct combine {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l;
		}
	};
	struct assign : matrix_oper {
		template<class lv, class rv> __BCinline__ auto& operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) = r);
		}
	};

	struct add_assign : matrix_oper, assignment {
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) += r);
		}
	};

	struct mul_assign : assignment {
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) *= r);
		}
	};

	struct sub_assign : matrix_oper, assignment {
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) -= r);
		}
	};

	struct div_assign : assignment {
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) /= r);
		}
	};
	struct alias_assign : assignment {
		template<class lv, class rv> __BCinline__ auto& operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) = r);
		}
	};

	struct alias_add_assign : assignment {
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) += r);
		}
	};

	struct alias_mul_assign : assignment {
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) *= r);
		}
	};

	struct alias_sub_assign : assignment {
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) -= r);
		}
	};

	struct alias_div_assign : assignment {
		template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
			return (const_cast<typename rm_const<lv&>::type>(l) /= r);
		}
	};
	struct equal {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l == r ? 1 : 0;
		}
	};

	struct greater {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l > r ? 1 : 0;
		}
	};
	struct lesser {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l < r ? 1 : 0;
		}
	};
	struct greater_equal {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l >= r ? 1 : 0;
		}
	};
	struct lesser_equal {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l <= r ? 1 : 0;
		}
	};
	struct max {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l > r ? l : r;
		}
	};
	struct min {
		template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
			return l < r ? l : r;
		}
	};
}
}



#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

