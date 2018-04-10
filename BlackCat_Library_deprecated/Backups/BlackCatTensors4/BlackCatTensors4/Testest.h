/*
 * Testest.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef TESTEST_H_
#define TESTEST_H_

namespace test {

	template<class T, class lv, class rv, class oper>
	struct binary_expression {
		lv left;
		rv right;

		binary_expression(lv l, rv r) : left(l), right(r) {}
	};


	template<class T, int ... dimensions>
	class Tensor_Math_Interface {
	};

	template<class T, int ... dimensions>
	class Vector;

	template<class T, int ... dimensions>
	class Matrix;
	// meta function type size of determines type
	template<class T, int... dimensions>
	struct shape_identity {

		template<int... dims>
		struct id {
			using type = Tensor_Math_Interface<T, dimensions...>;
		};
		template<int r>
		struct id<r> {
			using type = Vector<T, r>;
		};
		template<int r, int c>
		struct id<r,c > {
			using type = Matrix<T, r, c>;
		};
	};


	template<class T, int ... dimensions>
	class Tensor : shape_identity<T, dimensions...>::type {

	};

}

#endif /* TESTEST_H_ */
