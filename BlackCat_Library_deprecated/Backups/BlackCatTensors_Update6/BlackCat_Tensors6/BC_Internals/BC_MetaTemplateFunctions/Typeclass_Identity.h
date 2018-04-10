/*
 * BC_Tensor_Intermediate_Identity.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef TYPECLASS_IDENTITY_H_
#define TYPECLASS_IDENTITY_H_

#include "../BlackCat_Internal_ForwardDecls.h"

/*
 * Handles all template types.
 * Enables the superclass to return the derived type.
 */

namespace BC {
namespace BC_Shape_Identity {

	template<int...>
	struct ranks;

	template<class T, class ml, class dims, class LD = typename DEFAULT_LD<dims>::type>
	struct Identity;

	template<class T, class ml, template<int...> class shape, int ... dimensions, class LD>
	struct Identity<T, ml, shape<dimensions...>, LD> {
		using type = Tensor<T, ranks<dimensions...>, ml, LD>;
	};

	template<class T, class ml, template<int...> class shape, int rows, class LD>
	struct Identity<T, ml, shape<rows>, LD> {
		using type = Vector<T, rows, ml, LD>;
	};

	template<class T, class ml, template<int...> class shape, int rows, int cols, class LD>
	struct Identity<T, ml, shape<rows, cols>, LD> {
		using type = Matrix<T,rows, cols, ml, LD>;
	};

	template<class T, class ml, template<int...> class shape, int rows, int cols, int depth, class LD>
	struct Identity<T, ml, shape<rows, cols, depth>, LD> {
		using type = Cube<T, rows, cols, depth, ml, LD>;
	};
}

namespace BC_Evaluation_Identity {

	template<class T>
	struct asRef {
		using type = T&;
	};
	template<class T, class ml,class shape, class LD = typename DEFAULT_LD<shape>::type>
	struct Identity;

	template<class T, class ml, template<int...> class shape,int ... dimensions, class LD>
	struct Identity<T, ml, shape<dimensions...>, LD> {
		//if type is non-expression the type returns itself as a reference
		using nonRef_type = typename BC_Shape_Identity::Identity<T, ml, shape<dimensions...>, LD>::type;
		using type = typename asRef<nonRef_type>::type;
	};

	//if type is a nested type (type of type) we recursively search for the final non-list type class (type T)
	//Given that BlackCat_Tensors is designed so all general_functor_types (type T) are the first template parameter
	//we can find an apropriate "T" type to evaluate to. (aka an array_type)
	template<template<class...> class list, class T, class... set, class ml, template<int...> class shape, int ... dimensions, class LD>
	struct Identity<list<T, set...>, ml,shape<dimensions...>, LD> {
		using nonRef_type = typename BC_Shape_Identity::Identity<T, ml, shape<dimensions...>, LD>::type;
		using type = nonRef_type;
	};

}



namespace BC_ArrayType {

	template<class T, class voider = void>
	struct Identity {
		using type = T;
	};
	template<class T, class deriv>
	struct Identity<expression<T, deriv>> {
		using type = T;
	};
}


namespace BC_Substitute_Type {
	template<int...>
	struct ranks;

	template<class substitute_type, class>
	struct Identity;

	template<class sub_type, class T, class ml, int rows>
	struct Identity<sub_type, Vector<T, rows, ml>> {
		using type = Vector<sub_type, rows, ml>;
	};

	template<class sub_type, class T, class ml, int rows, int cols>
	struct Identity<sub_type, Matrix<T, rows, cols, ml>> {
		using type = Matrix<sub_type, rows, cols, ml>;
	};

	template<class sub_type, class T, class ml, int rows, int cols, int depth>
	struct Identity<sub_type, Cube<T, rows, cols, depth, ml>> {
		using type = Cube<sub_type, rows, cols, depth, ml>;
	};

	template<class sub_type, class T, class ml, template<int...> class shape, int... set>
	struct Identity<sub_type, Tensor<T, shape<set...>, ml>> {
		using type = Tensor<sub_type, shape<set...>, ml>;
	};
}
}

namespace BC_Derived_Rank {
	template<class> struct Identity;

	template<template< class, int, class, class> class tensor,
	class T, int row, class lib, class LD>
	struct Identity<tensor<T, row, lib, LD>> {
		static constexpr int value = 1;
	};

	template<template< class, int, int, class, class> class tensor,
	class T, int row, int col, class lib, class LD>
	struct Identity<tensor<T, row, col, lib, LD>> {
		static constexpr int value = 2;
	};;

	template<template< class, int, int, int, class, class> class tensor,
	class T, int row, int col, int depth, class lib, class LD>
	struct Identity<tensor<T, row, col,depth, lib, LD>> {
		static constexpr int value = 3;
	};;
}

#endif /* TYPECLASS_IDENTITY_H_ */
