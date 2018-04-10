/*
 * BC_Tensor_Super_Ace.h
 *
 *  Created on: Nov 18, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_INHERITLV1_SHAPE_H_
#define BC_TENSOR_INHERITLV1_SHAPE_H_

#include "BC_Tensor_SupplmentalLv1_Shape_impl.h"
#include "BC_MetaTemplate_UtilityMethods.h"
/*
 *
 * Compile time management of methods related to dimensionality
 *
 */

template<int... dims>
struct rank;

template<int... dimensions>
struct leading_dimension_generator;

template<int first, int... dimensions>
struct leading_dimension_generator<first, dimensions...> {

	using type = typename BC_MTF::extract_to<rank<first>, leading_dimension_generator<dimensions...>>::type;//<typename leading_dimension_generator<dimensions...>::type;
	static constexpr int value = first * leading_dimension_generator<dimensions...>::value;
};

template<int first>
struct leading_dimension_generator<first>  {
	using type = rank<first>;
	static constexpr int value = first;
};

template<int ... dimensions>
struct Shape {

	constexpr int size()  			const { return BC_Shape_Identity_impl::size<dimensions...>();  }
	constexpr int rows()  			const { return BC_Shape_Identity_impl::row<dimensions...>();   }
	constexpr int cols()  			const { return BC_Shape_Identity_impl::col<dimensions...>();   }
	constexpr int depth() 			const { return BC_Shape_Identity_impl::depth<dimensions...>(); }
	constexpr int pages() 			const { return BC_Shape_Identity_impl::pages<dimensions...>(); }
	constexpr int books() 			const { return BC_Shape_Identity_impl::books<dimensions...>(); }
	constexpr int libraries()	 	const { return BC_Shape_Identity_impl::libraries<dimensions...>(); }

	template<int dim_index>
	constexpr int dimension() 		const { return BC_Shape_Identity_impl::dimension<dim_index, dimensions...>(); }

	int dimension(int index) 		const { return BC_Shape_Identity_impl::libraries<index, dimensions...>(); }
};

#endif /* BC_TENSOR_INHERITLV1_SHAPE_H_ */
