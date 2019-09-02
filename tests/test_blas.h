/*
 * test_blas_validity.h
 *
 *  Created on: Feb 6, 2019
 *      Author: joseph
 */

#ifndef TEST_BLAS_VALIDITY_H_
#define TEST_BLAS_VALIDITY_H_

#include "test_common.h"

#if __has_include(<cblas.h>)
#include "cblas.h"
#elif __has_include(<mkl.h>)
#include <mkl.h>
#endif

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

	using BC::tensors::all;

	BC_TEST_BODY_HEAD

	using allocator_t = allocator<value_type>;
	using system_tag = typename allocator_traits<allocator_t>::system_tag;
	using compare_allocator = BC::Allocator<system_tag, value_type>;
	using blas = BC::blas::implementation<system_tag>;

	using mat = BC::Matrix<value_type, allocator_t>;
	using scalar = BC::Scalar<value_type, allocator_t>;

	using default_mat = BC::Matrix<value_type, compare_allocator>;

	BC::Stream<system_tag> stream;

	//The default allocator uses a global allocator, we free it to ensure that the next computations do not
	//Use any temporaries
	stream.get_allocator().free();

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

		stream.set_blas_pointer_mode_device();
		blas::gemm(stream, false, false,  sz, sz, sz,
				one.data(), h_a.data(), sz,
				h_b.data(), sz,
				zero.data(), h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y[0].zero();
		h_y[0].zero();

		y[0] = a * b[0];

		stream.set_blas_pointer_mode_device();
		blas::gemv(stream, false, sz, sz,
				one.data(), h_a.data(), sz,
				h_b.data(), 1,
				zero.data(), h_y.data(), 1);

			return BC::tensors::all(h_y[0].approx_equal(y[0]));
	)

	BC_TEST_DEF(
		y.zero();
		h_y.zero();
		y = a[0] * b[0].t();

		stream.set_blas_pointer_mode_device();
		blas::ger(stream,  sz, sz,
				one.data(), h_a[0].data(), 1,
				h_b[0].data(), 1,
				h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
			//test dot
		y[0][0] = a[0] * b[0];

		return BC::tensors::all(y[0][0].approx_equal(value_sum(a[0] % b[0])));
	)

	//scalar left test -------------------
	BC_TEST_DEF(
		y = two * a * b;

		stream.set_blas_pointer_mode_device();
		blas::gemm(stream, false, false,  sz, sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), sz,
				zero.data(), h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y[0].zero();
		h_y[0].zero();

		y[0] = two * a * b[0];

		stream.set_blas_pointer_mode_device();
		blas::gemv(stream, false,  sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), 1,
				zero.data(), h_y.data(), 1);

			return BC::tensors::all(h_y[0].approx_equal(y[0]));
	)

	BC_TEST_DEF(
		y.zero();
		h_y.zero();
		y = two * a[0] * b[0].t();

		stream.set_blas_pointer_mode_device();
		blas::ger(stream,  sz, sz,
				two.data(), h_a[0].data(), 1,
				h_b[0].data(), 1,
				h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
			//test dot
		y[0][0] = a[0] * b[0];

		return BC::tensors::all(y[0][0].approx_equal(value_sum(a[0] % b[0])));
	)

	// scalar right test -------------------------------------------
	BC_TEST_DEF(
		y =  a * two * b;

		stream.set_blas_pointer_mode_device();
		blas::gemm(stream, false, false,  sz, sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), sz,
				zero.data(), h_y.data(), sz);

		return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y[0].zero();
		h_y[0].zero();

		y[0] =   a * two * b[0];

		stream.set_blas_pointer_mode_device();
		blas::gemv(stream, false,  sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), 1,
				zero.data(), h_y.data(), 1);

			return BC::tensors::all(h_y[0].approx_equal(y[0]));
	)

	BC_TEST_DEF(
		y.zero();
		h_y.zero();
		y =  a[0]* two * b[0].t();

		stream.set_blas_pointer_mode_device();
		blas::ger(stream,  sz, sz,
				two.data(), h_a[0].data(), 1,
				h_b[0].data(), 1,
				h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
			//test dot
		y[0][0] = a[0] * b[0];

		return BC::tensors::all(y[0][0].approx_equal(value_sum(a[0] % b[0])));
	)

	//--------------------------------------------------------------
		stream.get_allocator().reserve(sz*sz*2 * sizeof(value_type));

	BC_TEST_DEF(
		y = a * b + a * b;

		stream.set_blas_pointer_mode_device();
		blas::gemm(stream, false, false,  sz, sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), sz,
				zero.data(), h_y.data(), sz);

		return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 1;
		y += a * b + a * b;

		stream.set_blas_pointer_mode_device();
		blas::gemm(stream, false, false,  sz, sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), sz,
				one.data(), h_y.data(), sz);

		return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 1;
		y -= a * b - a * b;
		return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 2;
		y += a * b + a * b + 1;

		stream.set_blas_pointer_mode_device();
		blas::gemm(stream, false, false,  sz, sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), sz,
				one.data(), h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 2;
		y -= a * b - a * b - 1;
		return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 2;
		y += (a * b) / (a * b);

		return BC::tensors::all(y.approx_equal(h_y));
	)

	BC_TEST_DEF(
		y = 1;
		h_y = 0;
		y -= (a * b) / (a * b);

		return BC::tensors::all(y.approx_equal(h_y));
	)

	//------------------------ Same blas tests as above but with scalar allocated on the stack ----------//
	//scalar left test -------------------
	BC_TEST_DEF(
		y = 2 * a * b;

		stream.set_blas_pointer_mode_device();
		blas::gemm(stream, false, false,  sz, sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), sz,
				zero.data(), h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y[0].zero();
		h_y[0].zero();

		y[0] = 2 * a * b[0];

		stream.set_blas_pointer_mode_device();
		blas::gemv(stream, false,  sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), 1,
				zero.data(), h_y.data(), 1);

			return BC::tensors::all(h_y[0].approx_equal(y[0]));
	)

	BC_TEST_DEF(
		y.zero();
		h_y.zero();
		y = 2 * a[0] * b[0].t();

		stream.set_blas_pointer_mode_device();
		blas::ger(stream,  sz, sz,
				two.data(), h_a[0].data(), 1,
				h_b[0].data(), 1,
				h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
			//test dot
		y[0][0] = a[0] * b[0];

		return BC::tensors::all(y[0][0].approx_equal(value_sum(a[0] % b[0])));
	)

	// scalar right test -------------------------------------------
	BC_TEST_DEF(
		y =  a * 2 * b;

		stream.set_blas_pointer_mode_device();
		blas::gemm(stream, false, false,  sz, sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), sz,
				zero.data(), h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_DEF(
		y[0].zero();
		h_y[0].zero();

		y[0] =   a * 2 * b[0];

		stream.set_blas_pointer_mode_device();
		blas::gemv(stream, false,  sz, sz,
				two.data(), h_a.data(), sz,
				h_b.data(), 1,
				zero.data(), h_y.data(), 1);

			return BC::tensors::all(h_y[0].approx_equal(y[0]));
	)

	BC_TEST_DEF(
		y.zero();
		h_y.zero();
		y =  a[0]* 2 * b[0].t();

		stream.set_blas_pointer_mode_device();
		blas::ger(stream,  sz, sz,
				two.data(), h_a[0].data(), 1,
				h_b[0].data(), 1,
				h_y.data(), sz);

			return BC::tensors::all(h_y.approx_equal(y));
	)

	BC_TEST_BODY_TAIL
}
}
}


#endif /* TEST_BLAS_VALIDITY_H_ */ 
