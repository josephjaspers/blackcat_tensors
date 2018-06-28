/*
 * BC_Expression_Binary_Functors.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifndef EXPRESSION_UNARY_FUNCTORS_H_
#define EXPRESSION_UNARY_FUNCTORS_H_

namespace BC {
namespace oper {
	struct negation {
		template<class lv> __BCinline__ auto operator ()(lv val) const {
			return -val;
		}
	};
	struct abs {
		template<class lv> __BCinline__ auto operator ()(lv val) const {
			return val < 0  ? -val : val;
		}
	};
	struct logical {
		template<class lv> __BCinline__ auto operator ()(lv val) const {
			return val == 0 ? 0 : 1;
		}
	};

	struct zero {
		template<class lv> __BCinline__ auto& operator ()(lv& val) const {
			return val = 0;
		}
	};
	struct one {
		template<class lv> __BCinline__ auto& operator ()(lv& val) const {
			return val = 1;
		}
	};
	struct fix {
		template<class lv> __BCinline__ auto& operator ()(lv& val) const {
			return isnan(val) || isinf(val) ? val = 0 : val;
		}
	};


	template<class scalar>
	struct norm {
		scalar min;
		scalar max;

		norm(scalar min_, scalar max_) : min(min_), max(max_) {}

		auto operator () (scalar v) const {
			return (v - min) / (max - min);
		}
	};

}
}


#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

