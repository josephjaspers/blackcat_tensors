/*
 * test_accessors.h
 *
 *  Created on: Dec 16, 2018
 *      Author: joseph
 */

#ifndef TEST_ACCESSORS_H_
#define TEST_ACCESSORS_H_

#include "test_common.h"

namespace BC {
namespace tests {

template<class scalar_t, template<class> class allocator>
int test_accessors(int sz=128) {

	BC_TEST_BODY_HEAD

	using mat = BC::Matrix<scalar_t, allocator<scalar_t>>;
	using vec = BC::Vector<scalar_t, allocator<scalar_t>>;
	using bmat = BC::Matrix<bool, allocator<bool>>;


	mat a(sz,sz);
	bmat validation(sz,sz);

	for (int i = 0; i < sz*sz; ++i) {
		a(i) = i;
	}





	//test slice
	BC_TEST_DEF(
		vec a0(a[0]);
		vec a1(a[1]);

		validation = a[0].approx_equal(a0) && a[1].approx_equal(a1);
		return BC::all(validation);
	)

	//test ranged slice
	BC_TEST_DEF(
		vec a0(a[0]);
		vec a1(a[1]);
		vec a2(a[2]);

		auto slice_range = a.slice(0, 3);

		bool ensure_correct_size = slice_range.size() == sz * 3;
		bool ensure_correct_cols = slice_range.cols() == 3;
		bool ensure_correct_rows = a.rows() == sz;

		validation = slice_range[0].approx_equal(a0) && slice_range[1].approx_equal(a1) && slice_range[2].approx_equal(a2);
		return BC::all(validation) && ensure_correct_size && ensure_correct_cols && ensure_correct_rows;
	)

	//test ranged slice
	BC_TEST_DEF(
		vec a0(a[1]);
		vec a1(a[2]);
		vec a2(a[3]);

		auto slice_range = a.slice(1, 4);

		bool ensure_correct_size = slice_range.size() == sz * 3;
		bool ensure_correct_cols = slice_range.cols() == 3;
		bool ensure_correct_rows = a.rows() == sz;

		validation = slice_range[0].approx_equal(a0) && slice_range[1].approx_equal(a1) && slice_range[2].approx_equal(a2);
		return BC::all(validation) && ensure_correct_size && ensure_correct_cols && ensure_correct_rows;
	)
	//test chunk
	BC_TEST_DEF(

		bmat validationmat(4,1);
		auto validation = validationmat[0];

		vec a1(a[1]);
		vec a2(a[2]);
		vec a3(a[3]);

		auto block_of_a = chunk(a,1,1)(4,3); //a 3x3 matrix starting at point 1,1


		bool ensure_correct_size = block_of_a.size() == 4 * 3;
		bool ensure_correct_rows = block_of_a.rows() == 4;
		bool ensure_correct_cols = block_of_a.cols() == 3;

		validation =
				block_of_a[0].approx_equal(a1.slice(1, 5)) &&
				block_of_a[1].approx_equal(a2.slice(1, 5)) &&
				block_of_a[2].approx_equal(a3.slice(1, 5));

		return BC::all(validation) && ensure_correct_size && ensure_correct_cols && ensure_correct_rows;
	)


	BC_TEST_BODY_TAIL
}
}
}



#endif /* TEST_ACCESSORS_H_ */
