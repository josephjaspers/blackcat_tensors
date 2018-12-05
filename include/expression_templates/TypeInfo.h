/*
 * TypeInfo.h
 *
 *  Created on: Dec 3, 2018
 *      Author: joseph
 */

#ifndef TYPEINFO_H_
#define TYPEINFO_H_

namespace BC {
namespace et {
template<class...> struct typeinfo;

template<class Scalar, class Allocator, class System_Tag>
struct typeinfo<Scalar, Allocator, System_Tag> {
	using scalar_t = Scalar;
	using allocator_t = Allocator;
	using system_tag = System_Tag;
};

template<class T>
struct typeinfo<T> : typeinfo<typename  T::scalar_t, typename T::allocator_, typename T::system_tag> {};

}
}



#endif /* TYPEINFO_H_ */
