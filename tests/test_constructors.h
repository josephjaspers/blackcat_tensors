/*
 * test_constructors.h
 *
 *  Created on: Dec 8, 2018
 *      Author: joseph
 */

#ifndef BC_TEST_CONSTRUCTORS_H_
#define BC_TEST_CONSTRUCTORS_H_

#include "test_common.h"

namespace BC {
namespace tests {

template<class value_type, template<class> class allocator>
int test_constructors(int sz=128) {

	using BC::tensors::all;

	using alloc_t = allocator<value_type>;
	using scal = BC::Scalar<value_type, alloc_t>;
	using vec = BC::Vector<value_type, alloc_t>;
	using mat = BC::Matrix<value_type, alloc_t>;
	using cube = BC::Cube<value_type, alloc_t>;

	BC_TEST_BODY_HEAD
	//Need to cover
	//Default
	//Standard
	//Copy
	//Move
	//Copy-oper
	//Move-Oper

	//-----------------------------------Default Constructor-----------------------------//
	BC_TEST_DEF(
		mat a;
		vec b;
		scal c;
		cube d;

		//Vectors .col() always return 1 even if uninitialized.
		bool ensure_size = a.size() == 0 && b.size() == 0 && c.size() == 1 && d.size() == 0;
		bool ensure_rows = a.rows() == 0 && b.rows() == 0 && c.rows() == 1 && d.rows() == 0;
		bool ensure_cols = a.cols() == 0 && b.cols() == 1 && c.cols() == 1 && d.cols() == 0;

		return ensure_size and ensure_rows and ensure_cols;
	)
	//-----------------------------------Standard Constructor-----------------------------//
	BC_TEST_DEF(
		mat a(5,7);
		return a.rows() == 5 && a.cols() == 7 && a.size() == 5*7;
	)
	BC_TEST_DEF(
		vec a(5);
		return a.rows() == 5 && a.cols() == 1 && a.size() == 5;
	)
	BC_TEST_DEF(
		cube a(5,7, 8);
		return a.rows() == 5 && a.cols() == 7 && a.size() == 5*7*8 && a.dimension(2) == 8; //dimension is 0 base
	)

	//-----------------------------------Copy Constructor-----------------------------//
	BC_TEST_DEF(
		mat a(5,5); a.rand(0, 10);
		mat b(a);

		return BC::tensors::all(b.approx_equal(a)) && b.rows() == 5 && b.cols() == 5 && a.data() != b.data();
	)
	//-----------------------------------Move Constructor-----------------------------//
	BC_TEST_DEF(
		mat a(5,5); a.rand(0, 10);

		auto* original_ptr = a.data();
		mat b(std::move(a));

		bool ensure_move = b.data() == original_ptr;
		bool ensure_diff = a.data() != original_ptr;
		bool ensure_swap_dims = true; //a.rows() ==0 && a.cols() ==0; //No longer guaranteed

		return ensure_move && ensure_diff && ensure_swap_dims;
	)
	//-----------------------------------Copy Oper-----------------------------//
	BC_TEST_DEF(
		mat a(5,5); a.rand(0, 10);
		mat b(5,5);

		b = a;
		return BC::tensors::all(b.approx_equal(a));
	)
//	//-----------------------------------Move Oper-----------------------------//
	BC_TEST_DEF(
		mat a(5,5); a.rand(0, 10);
		mat c(a); //copy to compare
		mat b;

		auto* original_ptr = a.data();
		b = std::move(a);

		return BC::tensors::all(b.approx_equal(c)) && a.data() != b.data() && b.data() == original_ptr;
	)

	BC_TEST_BODY_TAIL
}

}
}

#endif /* TEST_CONSTRUCTORS_H_ */
