/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_UNARY_FUNCTORS_H_
#define EXPRESSION_UNARY_FUNCTORS_H_

#include <cmath>
#include "tags.h"

namespace bc {
namespace oper {

struct Negation {
	template<class Value>
	BCINLINE Value operator ()(Value value) const {
		return -value;
	}

	template<class Value>
	BCINLINE static Value apply(Value value) {
		return -value;
	}
} negation;

}
}

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

