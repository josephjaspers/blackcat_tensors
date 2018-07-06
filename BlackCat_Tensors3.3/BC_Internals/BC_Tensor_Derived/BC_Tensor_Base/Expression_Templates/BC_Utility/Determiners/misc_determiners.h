///*
// * isPrimaryArray.h
// *
// *  Created on: May 20, 2018
// *      Author: joseph
// */
//
//#ifndef ISPRIMARYCORE_H_
//#define ISPRIMARYCORE_H_
//
//#include <type_traits>
//
//namespace BC {
//
//
//namespace internal {
//template<int, class, class> class Array;
//}
//
//template<class T> struct isPrimaryArray { static constexpr bool conditional = false; };
//template<int d, class T, class ml> struct isPrimaryArray<internal::Array<d,T,ml>> { static constexpr bool conditional = true; };
//template<class T> static constexpr bool is_array_core() { return isPrimaryArray<T>::conditional; }
//}
//
//
//
//#endif /* ISPRIMARYCORE_H_ */
