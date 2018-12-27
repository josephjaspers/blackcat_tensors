///*
// * Array_Packed_Triangle.h
// *
// *  Created on: Oct 14, 2018
// *      Author: joseph
// */
//
//#ifndef ARRAY_PACKED_TRIANGLE_H_
//#define ARRAY_PACKED_TRIANGLE_H_
//
//#include "Array_Base.h"
//
//namespace BC {
//namespace et     {
//
//enum upLo {
//    up = 1,
//    lo = 0
//};
//
//template<upLo upper, class value_type_, class allocator_t_>
//struct Array_Packed_Triangle
//        : Array_Base<Array_Packed_Triangle<upLo, value_type_, allocator_t_>, 2>,
//          Shape<2> {
//
//    static constexpr BC::size_t  DIMS     { return 2; }
//    static constexpr BC::size_t  ITERATOR { return 2; }
//
//    using value_type = value_type_;
//    using allocator_t = allocator_t_;
//
//    value_type* array;
//
//    Array_Packed_Triangle(int N) : Shape<2>(N, N) {
//        allocator_t::allocate(array, this->size() / 2);
//    }
////I would like to thank stack overflow for its contribution
//    value_type& operator () (int row, BC::size_t  col) {
//
//    }
//    const value_type& operator () (int row, BC::size_t  col) const {
//
//    }
//
//};
//
//
//
//}
//}
//
//
//
//#endif /* ARRAY_PACKED_TRIANGLE_H_ */
