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
//template<upLo upper, class scalar_t_, class allocator_t_>
//struct Array_Packed_Triangle
//        : Array_Base<Array_Packed_Triangle<upLo, scalar_t_, allocator_t_>, 2>,
//          Shape<2> {
//
//    static constexpr int DIMS()     { return 2; }
//    static constexpr int ITERATOR() { return 2; }
//
//    using scalar_t = scalar_t_;
//    using allocator_t = allocator_t_;
//
//    scalar_t* array;
//
//    Array_Packed_Triangle(int N) : Shape<2>(N, N) {
//        allocator_t::allocate(array, this->size() / 2);
//    }
////I would like to thank stack overflow for its contribution
//    scalar_t& operator () (int row, int col) {
//
//    }
//    const scalar_t& operator () (int row, int col) const {
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
