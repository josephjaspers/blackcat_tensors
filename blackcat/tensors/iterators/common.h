/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef STL_tensor_iterator_dimension_COMMON_H_
#define STL_tensor_iterator_dimension_COMMON_H_

#include <iterator>

namespace bc {
namespace tensors {
namespace iterators {

enum direction {
	forward = 1,
	reverse = -1
};

enum initpos {
	start=0,
	end=1
};

template<class T>
struct iterator_traits: std::iterator_traits<T> {

	using system_tag =
			bc::traits::conditional_detected_t<
			bc::traits::query_system_tag, T, host_tag>;
};



}
}
}



#endif /* STL_tensor_iterator_dimension_COMMON_H_ */
