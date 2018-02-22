///*
// * BC_Tensor_Super_Ace.h
// *
// *  Created on: Nov 18, 2017
// *      Author: joseph
// */
//
//#ifndef STATIC_SHAPE_H_
//#define STATIC_SHAPE_H_
//
//#include "../BlackCat_Internal_GlobalUnifier.h"
//#include "Static_Shape_Impl.h"
//#include "Shape_DefaultLD_Impl.h"
//#include <type_traits>
//#include <vector>
///*
// * Compile time management of methods related to dimensionality
// */
//namespace BC {
//
//	template<int>class Dynamic_Inner_Shape;
//	template<int>class Dynamic_Outer_Shape;
//	template<class>class DEFAULT_LD;
//
//template<class Inner_Shape, class Outer_Shape = typename DEFAULT_LD<Inner_Shape>::type>
//struct Tensor_Shape :
//	MTF::IF_ELSE<Inner_Shape::isDynamic, Dynamic_Inner_Shape<Inner_Shape::RANK   >, Inner_Shape>::type,
//	MTF::IF_ELSE<Outer_Shape::isDynamic, Dynamic_Outer_Shape<Outer_Shape::LD_RANK>, Outer_Shape>::type {
//private:
////	template<int> struct MAKE_ILLEGAL {};
////	template<class, class> struct INVALIDATE {};
//public:
//	static constexpr bool LDisDefault = MTF::is_same<Outer_Shape, typename DEFAULT_LD<Inner_Shape>::type>::conditional;
//	static constexpr bool Dynamic_Inner = Inner_Shape::isDynamic;
//	static constexpr bool Dynamic_Outer = Outer_Shape::isDynamic;
////	static constexpr int  RANK = Inner_Shape::RANK;
////	static constexpr int  SIZE = Inner_Shape::size();
//
//	using outer_shape = Outer_Shape;
//	using inner_shape = Inner_Shape;
//
////	//IF BOTH DYNAMIC (AND DEFAULT LD)
////	using default_initializer = typename MTF::IF_ELSE<MTF::AND<LDisDefault, Dynamic_Outer>::conditional, std::initializer_list<int>, MAKE_ILLEGAL<0>>::type;
////	Tensor_Shape(default_initializer inner_dims) : Inner_Shape(inner_dims), Outer_Shape(inner_dims) {}
////
////	//IF INNER IS DYNAMIC OUTER STATIC
////	using inner_exclusive_initializer = typename MTF::IF_ELSE<MTF::AND<Dynamic_Inner,!Dynamic_Outer>::conditional, std::initializer_list<int>, MAKE_ILLEGAL<3>>::type;
////	Tensor_Shape(inner_exclusive_initializer inner_dims) : Inner_Shape(inner_dims) {}
////
////	//IF OUTER IS DYNAMIC INNER STATIC
////	using outer_exclusive_initializer = typename MTF::IF_ELSE<MTF::AND<Dynamic_Inner,!Dynamic_Outer>::conditional, int*, MAKE_ILLEGAL<4>>::type;
////	Tensor_Shape(outer_exclusive_initializer outer_dims) : Outer_Shape(outer_dims) {}
////
////	//IF BOTH DYNAMIC (AND NOT DEFAULT LD)
////	using inner_initializer = typename MTF::IF_ELSE<MTF::AND<!LDisDefault, Dynamic_Inner>::conditional, std::initializer_list<int>, MAKE_ILLEGAL<1>>::type;
////	using outer_initializer = typename MTF::IF_ELSE<MTF::AND<!LDisDefault, Dynamic_Outer>::conditional, int*, MAKE_ILLEGAL<2>>::type;
////	Tensor_Shape(inner_initializer inner, outer_initializer outer) : Inner_Shape(inner), Outer_Shape(outer) {}
////
////	//IF BOTH STATIC (THROWS AN EXCEPTION IF CALLED ILLEGALY)
////	Tensor_Shape() = default;
////
////	//SAME TYPE BUT ACCEPTS ARBITRARY INTS
////	Tensor_Shape(default_initializer inner_dims, int) : Inner_Shape(inner_dims), Outer_Shape(inner_dims) {}
////	Tensor_Shape(inner_exclusive_initializer inner_dims, int) : Inner_Shape(inner_dims) {}
////	Tensor_Shape(int, outer_exclusive_initializer outer_dims) : Outer_Shape(outer_dims) {}
////	Tensor_Shape(int, int)  {}
////
////
////	//Inner Dynamic_ Outer_Not
////	template<class in, class ou>
////	Tensor_Shape(MTF::IF_ELSE<MTF::AND<in::isDynamic, !ou::isDynamic>::conditional, const Tensor_Shape<in, ou>&, INVALIDATE<in, ou>> param)
////	: Inner_Shape(param.getShape()){}
////
////	//Outer Dynamic
////	template<class in, class ou>
////	Tensor_Shape(MTF::IF_ELSE<MTF::AND<!in::isDynamic, ou::isDynamic>::conditional, const Tensor_Shape<in, ou>&, INVALIDATE<in, ou>> param)
////		: Outer_Shape(param.getOuterShape()){}
////
////
////	//Both Dynamic
////	template<class in, class ou>
////	Tensor_Shape(MTF::IF_ELSE<MTF::AND<in::isDynamic, ou::isDynamic>::conditional, const Tensor_Shape<in, ou>&, INVALIDATE<in, ou>> param)
////	: Inner_Shape(param.getShape()), Outer_Shape(param.getOuterShape()) {}
//
//};
//
//
//		namespace template_to_array {
//
//			template<class >
//			struct f;
//
//			template<int ... set, template<int...> class list, int front>
//			struct f<list<front, set...>> {
//				static void fill(int* ary) {
//					ary[0] = front;
//					f<list<set...>>::fill(&ary[1]);
//				}
//			};
//			template<template<int...> class list, int front>
//			struct f<list<front>> {
//				static void fill(int* ary) {
//					ary[0] = front;
//				}
//			};
//		}
//
//
//template<int ... dimensions>
//struct Inner_Shape {
//
//	static constexpr int RANK = sizeof...(dimensions);
//	static constexpr bool isDynamic = MTF::sum<dimensions...>::value == 0;
//
//	void printDimensions() const { BC_Shape_Identity_impl::print<Inner_Shape<dimensions...>>(); }
//	int passInner() { return 0; }
//
//	const auto getShape() const {
//		std::vector<int> sh(RANK);
//		template_to_array::f<Inner_Shape<dimensions...>>::fill(&sh[0]);
//		  return sh;
//	}
//	constexpr int order() const { return sizeof...(dimensions); }
//	constexpr int size()  const { return BC_Shape_Identity_impl::size<dimensions...>();  }
//	constexpr int rows()  const { return BC_Shape_Identity_impl::row<dimensions...>();   }
//	constexpr int cols()  const { return BC_Shape_Identity_impl::col<dimensions...>();   }
//	constexpr int depth() const { return BC_Shape_Identity_impl::depth<dimensions...>(); }
//	constexpr int pages() const { return BC_Shape_Identity_impl::pages<dimensions...>(); }
//	constexpr int books() const { return BC_Shape_Identity_impl::books<dimensions...>(); }
//	constexpr int libraries() const { return BC_Shape_Identity_impl::libraries<dimensions...>(); }
//
//	template<int dim_index> constexpr int dimension() const { return BC_Shape_Identity_impl::dimension<dim_index, dimensions...>(); }
//};
//
//template<int... dimensions>
//struct Outer_Shape {
//
//	static constexpr int LD_RANK = sizeof...(dimensions);
//	static constexpr bool isDynamic = MTF::sum<dimensions...>::value == 0;
//
//	void printLeadingDimensions() const { BC_Shape_Identity_impl::print<Outer_Shape<dimensions...>>(); }
//
//	const auto getOuterShape() const {
//		std::vector<int> sh(LD_RANK);
//		template_to_array::f<Outer_Shape<dimensions...>>::fill(&sh[0]);
//		  return sh;
//	}
//
//	constexpr int LD_size()  const { return BC_Shape_Identity_impl::size<dimensions...>();  }
//	constexpr int LD_rows()  const { return BC_Shape_Identity_impl::row<dimensions...>();   }
//	constexpr int LD_cols()  const { return BC_Shape_Identity_impl::col<dimensions...>();   }
//	constexpr int LD_depth() const { return BC_Shape_Identity_impl::depth<dimensions...>(); }
//	constexpr int LD_pages() const { return BC_Shape_Identity_impl::pages<dimensions...>(); }
//	constexpr int LD_books() const { return BC_Shape_Identity_impl::books<dimensions...>(); }
//	constexpr int LD_libraries() const { return BC_Shape_Identity_impl::libraries<dimensions...>(); }
//
//	template<int dim_index> constexpr int LD_dimension() const { return BC_Shape_Identity_impl::dimension<dim_index, dimensions...>(); }
//};
//
//
//}
//#endif /* STATIC_SHAPE_H_ */
