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

	using alloc_t = allocator<value_type>;
	using system_tag = typename BC::allocator_traits<alloc_t>::system_tag;

	if (std::is_same<system_tag, BC::device_tag>::value) {
		std::cout << "BLAS test limited CPU" << std::endl;
		return 0;
	}

#ifdef __CUDACC__
	using host_allocator = std::conditional_t<std::is_same<host_tag, system_tag>::value,
												BC::Basic_Allocator<value_type>,
												BC::Cuda_Managed<value_type>>;
#else
	using host_allocator = BC::Basic_Allocator<value_type>;
#endif

//	using vec = BC::Vector<value_type, alloc_t>;
	using mat = BC::Matrix<value_type, alloc_t>;
	using scalar = BC::Scalar<value_type, alloc_t>;

//	using host_vec = BC::Vector<value_type, host_allocator>;
	using host_mat = BC::Matrix<value_type, host_allocator>;

	//gets the 'host' (cpu) wrapper for BLAS calls.
	using blas = BC::blas::Host;

	//default cpu context (used in the BC-blas wrapper')
	//this should be ignored if it just boiler-plate
	BC::Context<BC::host_tag> context;

	scalar A(value_type(2));
	mat a(sz, sz);
	mat b(sz, sz);
	mat y(sz, sz);

//	value_type host_A = 2;
	host_mat h_a(sz, sz);
	host_mat h_b(sz, sz);
	host_mat h_y(sz, sz);

	a.randomize(0,10);
	b.randomize(0,10);
	y.randomize(0,10);

	h_a = a;
	h_b = b;
	h_y = y;

//	value_type one = 1;
//	value_type zero = 0;


	BC_TEST_DEF(
		y = a * b;

		blas::gemm(context, false, false,  sz, sz, sz,
				1, h_a.data(), sz,
				h_b.data(), sz,
				0, h_y.data(), sz);

			return BC::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = a * b + a * b;

		blas::gemm(context, false, false,  sz, sz, sz,
				2, h_a.data(), sz,
				h_b.data(), sz,
				0, h_y.data(), sz);

		return BC::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 1;
		y += a * b + a * b;

		blas::gemm(context, false, false,  sz, sz, sz,
				2, h_a.data(), sz,
				h_b.data(), sz,
				1, h_y.data(), sz);

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
				2, h_a.data(), sz,
				h_b.data(), sz,
				1, h_y.data(), sz);

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
