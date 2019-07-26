/*
 * Function_AutoBroadcast.h
 *
 *  Created on: Jul 26, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_EXPRS_FUNCTION_AUTOBROADCAST_H_
#define BLACKCAT_TENSORS_EXPRS_FUNCTION_AUTOBROADCAST_H_

namespace BC {
namespace tensors {
namespace exprs {

struct Auto_Broadcast {

	using index_aware_function = std::true_type;
	using auto_broadcast = std::true_type;

	template<class Expression> BCINLINE
	auto operator () (const Expression& expression, BC::size_t x, BC::size_t y) const {
		return expression(x % expression.dimension(0),
					y % expression.dimension(1));
	}
	template<class Expression> BCINLINE
	auto operator () (const Expression& expression,
			BC::size_t x, BC::size_t y, BC::size_t z) const {
		return expression(x % expression.dimension(0),
					y % expression.dimension(1),
					z % expression.dimension(2));

	}

	template<class Expression> BCINLINE
	auto operator () (const Expression& expression,
			BC::size_t x, BC::size_t y, BC::size_t z, BC::size_t a) const {
		return expression(x % expression.dimension(0),
					y % expression.dimension(1),
					z % expression.dimension(2),
					a % expression.dimension(3));
	}
	template<class Expression> BCINLINE
	auto operator () (const Expression& expression,
			BC::size_t x, BC::size_t y, BC::size_t z,
			BC::size_t a, BC::size_t b) const {
		return expression(x % expression.dimension(0),
					y % expression.dimension(1),
					z % expression.dimension(2),
					a % expression.dimension(3),
					b % expression.dimension(4));
	}


	template<class Expression, class Index> BCINLINE
	auto operator () (const Expression& expression, Index idx) const {
		return expression[idx % expression.size()];
	}


};



}
}
}



#endif /* FUNCTION_AUTOBROADCAST_H_ */
