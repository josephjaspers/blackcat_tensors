/*
 * BC_Tensor_Intermediate_Identity.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPPLMENTALLV2_IDENTITY_H_
#define BC_TENSOR_SUPPLMENTALLV2_IDENTITY_H_

/*
 * Compile Time Unit
 */




#include "BC_Internals_Include.h"
namespace BC_Shape_Identity {

	template<class T, class ml, int ... dimensions>
	struct Identity {
		using type = Tensor<T, ml, dimensions...>;
	};

	template<class T, class ml, int rows>
	struct Identity<T, ml, rows> {
		using type = Vector<T, ml, rows>;
	};

	template<class T, class ml, int rows, int cols>
	struct Identity<T, ml, rows, cols> {
		using type = Matrix<T, ml, rows, cols>;
	};

	template<class T, class ml, int rows, int cols, int depth>
	struct Identity<T, ml, rows, cols, depth> {
		using type = Cube<T, ml, rows, cols, depth>;
	};
}

namespace BC_Evaluation_Identity {

	template<class T>
	struct asRef {
		using type = T&;
	};

	template<class T, class ml, int ... dimensions>
	struct Identity {
		//if type is non-expression the type returns itself as a reference
		using nonRef_type = T;
		using type = typename asRef<nonRef_type>::type;
	};

	template<class T, class ml, int ... dimensions>
	struct Identity<expression<T>, ml, dimensions...> {
		using nonRef_type = typename BC_Shape_Identity::Identity<T, ml, dimensions...>::type;
		using type = nonRef_type;
	};
}

namespace BC_ArrayType {

	template<class T, class voider = void>
	struct Identity {
		using type = T;
	};
	template<class T>
	struct Identity<expression<T>> {
		using type = T;
	};
}
#endif /* BC_TENSOR_SUPPLMENTALLV2_IDENTITY_H_ */
