/*
 * Array_Reformat.h
 *
 *  Created on: Oct 5, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_FORMAT_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_FORMAT_H_

#include "Array_Base.h"
namespace BC {
namespace internal {

template<class PARENT>
struct Array_Format
		: Array_Base<Array_Format<PARENT>, PARENT::DIMS()>, Shape<PARENT::DIMS()> {

	using scalar_t = typename PARENT::scalar_t;
	using mathlib_t = typename PARENT::mathlib_t;

	__BCinline__ static constexpr int DIMS() 		 { return PARENT::DIMS(); }
	__BCinline__ static constexpr int ITERATOR()	 { return DIMS(); }

	scalar_t* array_slice;

	template<class... format, class enabler = std::enable_if_t<MTF::seq_of<int, format...>>> __BCinline__
	Array_Format(const scalar_t* array, PARENT parent_, format... integers)
	: Shape<PARENT::DIMS()> (parent_.as_shape()), array_slice(const_cast<scalar_t*>(array)) {
		static_assert(sizeof...(format) == DIMS() - 1, "FORMATTED DIMENSIONS MUST BE EQUAL TO NUMB_DIMENSION - 1. DIMENSION(1) CAN NOT BE REFORMATTED");
		auto format_array = BC::make_array(integers...);

		for (int i = 0; i < format_array.size(); ++i) {
			this->m_inner_shape[i] = parent_.dimension(format_array[i] - 1);
			this->m_outer_shape[i] = parent_.leading_dimension(format_array[i] - 1);

			std::cout << " ld is " << this->leading_dimension(i) << std::endl;
		}
	}

	__BCinline__ const scalar_t* memptr() const { return array_slice; }
	__BCinline__	   scalar_t* memptr()   	{ return array_slice; }

	};
}
}



#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_FORMAT_H_ */
