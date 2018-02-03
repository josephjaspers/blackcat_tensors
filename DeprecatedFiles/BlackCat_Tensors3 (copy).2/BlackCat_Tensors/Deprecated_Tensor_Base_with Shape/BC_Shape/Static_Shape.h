/*
 * BC_Tensor_Super_Ace.h
 *
 *  Created on: Nov 18, 2017
 *      Author: joseph
 */

#ifndef STATIC_SHAPE_H_
#define STATIC_SHAPE_H_

#include "../BlackCat_Internal_GlobalUnifier.h"
#include "Static_Shape_Impl.h"
#include "Shape_DefaultLD_Impl.h"
#include "template_to_array.h"
#include <type_traits>
#include <vector>

namespace BC {

template<int ... dimensions>
struct Outer_Shape {

	static constexpr int COMPILE_TIME_LD_ROWS() {return BC_Shape_Identity_impl::row<dimensions...>();}
	static constexpr int COMPILE_TIME_LD_COLS() {return BC_Shape_Identity_impl::col<dimensions...>();}
	static constexpr bool isDynamic = MTF::sum<dimensions...>::value == 0;
	static constexpr bool DynamicOuterShape() { return false; }
	static constexpr int  LD_RANK()  {return sizeof...(dimensions); }


	void printLeadingDimensions() const { BC_Shape_Identity_impl::print<Outer_Shape<dimensions...>>(); }

	std::vector<int> getLD() const {
		std::vector<int> sh(LD_RANK());
		template_to_array::f<Outer_Shape<dimensions...>>::fill(&sh[0]);
		return sh;
	}

	constexpr int LD_size() const  { return BC_Shape_Identity_impl::size<dimensions...>();  }
	constexpr int LD_rows() const  { return BC_Shape_Identity_impl::row<dimensions...>();   }
	constexpr int LD_cols() const  { return BC_Shape_Identity_impl::col<dimensions...>();   }
	constexpr int LD_depth() const { return BC_Shape_Identity_impl::depth<dimensions...>(); }
	constexpr int LD_pages() const { return BC_Shape_Identity_impl::pages<dimensions...>(); }
	constexpr int LD_books() const { return BC_Shape_Identity_impl::books<dimensions...>(); }
	constexpr int LD_libraries() const { return BC_Shape_Identity_impl::libraries<dimensions...>(); }
	template<int dim_index> constexpr int LD_dimension() const { return BC_Shape_Identity_impl::dimension<dim_index, dimensions...>(); }
};

template<int ... dimensions>
struct Inner_Shape {
	static constexpr int COMPILE_TIME_ROWS() { return BC_Shape_Identity_impl::row<dimensions...>();}
	static constexpr int COMPILE_TIME_COLS() { return BC_Shape_Identity_impl::col<dimensions...>();}
	static constexpr bool DynamicInnerShape() { return false; }
	static constexpr bool isDynamic = MTF::sum<dimensions...>::value == 0;
	static constexpr int RANK() { return  sizeof...(dimensions); }



	void printDimensions() const { BC_Shape_Identity_impl::print<Inner_Shape<dimensions...>>(); }

	std::vector<int> getShape() const {
		std::vector<int> sh(RANK());
		template_to_array::f<Inner_Shape<dimensions...>>::fill(&sh[0]);
		return sh;
	}

	constexpr int order() const { return sizeof...(dimensions); }
	constexpr int size() const { return BC_Shape_Identity_impl::size<dimensions...>(); }
	constexpr int rows() const { return BC_Shape_Identity_impl::row<dimensions...>(); }
	constexpr int cols() const { return BC_Shape_Identity_impl::col<dimensions...>(); }
	constexpr int depth() const { return BC_Shape_Identity_impl::depth<dimensions...>(); }
	constexpr int pages() const { return BC_Shape_Identity_impl::pages<dimensions...>(); }
	constexpr int books() const { return BC_Shape_Identity_impl::books<dimensions...>(); }
	constexpr int libraries() const { return BC_Shape_Identity_impl::libraries<dimensions...>();}
	template<int dim_index> constexpr int dimension() const { return BC_Shape_Identity_impl::dimension<dim_index, dimensions...>(); }

};
}
#endif /* STATIC_SHAPE_H_ */
