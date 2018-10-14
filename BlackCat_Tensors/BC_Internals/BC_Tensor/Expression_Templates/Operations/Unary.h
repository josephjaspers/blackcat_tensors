/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_UNARY_FUNCTORS_H_
#define EXPRESSION_UNARY_FUNCTORS_H_

namespace BC {
namespace internal {
namespace oper {
	struct negation {
		template<class lv> __BCinline__ lv operator ()(lv val) const {
			return -val;
		}
	};
	struct absolute {
		template<class lv> __BCinline__ lv operator ()(lv val) const {
			return abs(val);
		}
	};
	struct logical {
		template<class lv> __BCinline__ lv operator ()(lv val) const {
			return val == 0 ? 0 : 1;
		}
	};

	template<class scalar>
	struct norm {
		scalar min;
		scalar max;

		norm(scalar min_, scalar max_) : min(min_), max(max_) {}

		__BCinline__ auto operator () (scalar v) const {
			return (v - min) / (max - min);
		}
	};

//DEFINE FOR SHORTHAND GENERATION
#define BLACKCAT_FUNCTION(func)             						        \
struct func {                    								\
  template<class scalar_t> __BCinline__							\
   scalar_t operator () (scalar_t s) const {  return std::func(s); }			\
};

BLACKCAT_FUNCTION(acos)
BLACKCAT_FUNCTION(acosh)
BLACKCAT_FUNCTION(sin)
BLACKCAT_FUNCTION(asin)
BLACKCAT_FUNCTION(asinh)
BLACKCAT_FUNCTION(atan)
BLACKCAT_FUNCTION(atanh)
BLACKCAT_FUNCTION(cbrt)
BLACKCAT_FUNCTION(ceil)
BLACKCAT_FUNCTION(cos)
BLACKCAT_FUNCTION(cosh)
BLACKCAT_FUNCTION(exp)
BLACKCAT_FUNCTION(exp2)
BLACKCAT_FUNCTION(fabs)
BLACKCAT_FUNCTION(floor)
BLACKCAT_FUNCTION(fma)
BLACKCAT_FUNCTION(isinf)
BLACKCAT_FUNCTION(isnan)
BLACKCAT_FUNCTION(log)
BLACKCAT_FUNCTION(log2)
BLACKCAT_FUNCTION(lrint)
BLACKCAT_FUNCTION(lround)
BLACKCAT_FUNCTION(modf)
BLACKCAT_FUNCTION(sqrt)
BLACKCAT_FUNCTION(tan)
BLACKCAT_FUNCTION(tanh)
}
}
}

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

