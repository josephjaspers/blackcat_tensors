/*
 * test_blas_validity.h
 *
 *  Created on: Feb 6, 2019
 *      Author: joseph
 */

#ifndef TEST_BLAS_VALIDITY_H_
#define TEST_BLAS_VALIDITY_H_

#include "test_common.h"
#include "cblas.h"

namespace BC {
namespace tests {

/*
 * Need to cover...
 * =
 * +=
 * -=
 * \=
 * alias
 *
 * 'injectable' expressions
 * 'advanced' expressions
 */

template<class value_type, template<class> class allocator>
int test_blas(int sz=128) {

	BC_TEST_BODY_HEAD

	using allocator_t = allocator<value_type>;
	using system_tag = typename BC::allocator_traits<allocator_t>::system_tag;
	using compare_allocator = BC::allocator::implementation<system_tag, value_type>;
	using blas = BC::blas::implementation<system_tag>;

	using mat = BC::Matrix<value_type, allocator_t>;
	using scalar = BC::Scalar<value_type, allocator_t>;

	using default_mat = BC::Matrix<value_type, compare_allocator>;

	BC::Context<system_tag> context;

	//Initialize the memory pool
	context.get_allocator().reserve(sz*sz*2 * sizeof(value_type));

	scalar A(value_type(2));
	mat a(sz, sz);
	mat b(sz, sz);
	mat y(sz, sz);

	default_mat h_a(sz, sz);
	default_mat h_b(sz, sz);
	default_mat h_y(sz, sz);

	a.randomize(0,10);
	b.randomize(0,10);
	y.randomize(0,10);

	h_a = a;
	h_b = b;
	h_y = y;

	scalar one = 1;
	scalar two = 2;
	scalar zero = 0;

	BC_TEST_DEF(
		y = a * b;

		blas::gemm(context, false, false,  sz, sz, sz,
				one, h_a.data(), sz,
				h_b.data(), sz,
				zero, h_y.data(), sz);

			return BC::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = a * b + a * b;

		blas::gemm(context, false, false,  sz, sz, sz,
				two, h_a.data(), sz,
				h_b.data(), sz,
				zero, h_y.data(), sz);

		return BC::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 1;
		y += a * b + a * b;

		blas::gemm(context, false, false,  sz, sz, sz,
				two, h_a.data(), sz,
				h_b.data(), sz,
				one, h_y.data(), sz);

		return BC::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 1;
		y -= a * b - a * b;
		return BC::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 2;
		y += a * b + a * b + 1;

		blas::gemm(context, false, false,  sz, sz, sz,
				two, h_a.data(), sz,
				h_b.data(), sz,
				one, h_y.data(), sz);

			return BC::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 2;
		y -= a * b - a * b - 1;
		return BC::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 2;
		y += (a * b) / (a * b);

		return BC::all(y.approx_equal(h_y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 0;
		y -= (a * b) / (a * b);

		return BC::all(y.approx_equal(h_y));
	)

	BC_TEST_BODY_TAIL
}
}
}


#endif /* TEST_BLAS_VALIDITY_H_ */
