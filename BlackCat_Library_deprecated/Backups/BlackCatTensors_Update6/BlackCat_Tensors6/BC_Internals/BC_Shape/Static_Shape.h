/*
 * BC_Tensor_Super_Ace.h
 *
 *  Created on: Nov 18, 2017
 *      Author: joseph
 */

#ifndef STATIC_SHAPE_H_
#define STATIC_SHAPE_H_

#include "../BlackCat_Internal_GlobalUnifier.h"
#include <vector>

#include "../BC_Shape/Static_Shape_Inner_Impl.h"
#include "../BC_Shape/Static_Shape_Outer_Impl.h"
/*
 * Compile time management of methods related to dimensionality
 */
namespace BC {

	namespace printHelper { template<class> struct f; }

template<int ... dimensions>
struct Static_Inner_Shape {

	static constexpr int RANK = sizeof...(dimensions);
	constexpr bool inner_shape_fast_copy() const { return true; }
	void printDimensions() const { BC_Shape_Identity_impl::print<Static_Inner_Shape<dimensions...>>(); }

	constexpr int size()  const { return BC_Shape_Identity_impl::size<dimensions...>(); }
	constexpr int rows()  const { return BC_Shape_Identity_impl::row<dimensions...>();   }
	constexpr int cols()  const { return BC_Shape_Identity_impl::col<dimensions...>();   }
	constexpr int depth() const { return BC_Shape_Identity_impl::depth<dimensions...>(); }
	constexpr int pages() const { return BC_Shape_Identity_impl::pages<dimensions...>(); }
	constexpr int books() const { return BC_Shape_Identity_impl::books<dimensions...>(); }
	constexpr int libraries() const { return BC_Shape_Identity_impl::libraries<dimensions...>(); }

	auto getShape() const {
		std::vector<int> sh(RANK);
		sh[0];
		printHelper::f<Static_Inner_Shape>::fill(&sh[0]);
		return sh;
	}

	template<int dim_index> constexpr int dimension() const { return BC_Shape_Identity_impl::dimension<dim_index, dimensions...>(); }
};

template<int... dimensions>
struct Static_Outer_Shape {

	static constexpr int LD_RANK = sizeof...(dimensions);
	constexpr bool outer_shape_fast_copy() const { return false; }
	void printLeadingDimensions() const { BC_Shape_Identity_impl::print<Static_Outer_Shape<dimensions...>>(); }

	constexpr int LD_size()  const { return BC_Shape_Identity_impl::size<dimensions...>(); }
	constexpr int LD_rows()  const { return BC_Shape_Identity_impl::row<dimensions...>(); }
	constexpr int LD_cols()  const { return BC_Shape_Identity_impl::col<dimensions...>(); }
	constexpr int LD_depth() const { return BC_Shape_Identity_impl::depth<dimensions...>(); }
	constexpr int LD_pages() const { return BC_Shape_Identity_impl::pages<dimensions...>(); }
	constexpr int LD_books() const { return BC_Shape_Identity_impl::books<dimensions...>(); }
	constexpr int LD_libraries() const { return BC_Shape_Identity_impl::libraries<dimensions...>(); }

	auto getLDShape() const {
		std::vector<int> sh(LD_RANK);
		sh[0];
		printHelper::f<Static_Outer_Shape>::fill(&sh[0]);
		return sh;
	}

	template<int dim_index> constexpr int LD_dimension() const { return BC_Shape_Identity_impl::dimension<dim_index, dimensions...>(); }
};
namespace printHelper {

	template<class >
	struct f;

	template<int ... set, template<int...> class list, int front>
	struct f<list<front, set...>> {
		static void fill(int* ary) {
			ary[0] = front;
			f<list<set...>>::fill(&ary[1]);
		}
	};
	template<template<int...> class list, int front>
	struct f<list<front>> {
		static void fill(int* ary) {
			ary[0] = front;
		}
	};
}

}
#endif /* STATIC_SHAPE_H_ */
