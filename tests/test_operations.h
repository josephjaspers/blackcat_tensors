/*
 * blas_function_testing.h
 *
 *  Created on: Dec 1, 2018
 *      Author: joseph
 */

#ifndef BC_BLAS_Function_TESTING_H_
#define BC_BLAS_Function_TESTING_H_

#include "test_common.h"

namespace BC {
namespace tests {

template<class value_type, template<class> class allocator=BC::Basic_Allocator>
int test_operations(int sz=128) {

	using alloc_t = allocator<value_type>;
	using mat =  BC::Matrix<value_type, alloc_t>;
	using bmat = BC::Matrix<bool, allocator<bool>>;

	BC_TEST_BODY_HEAD

	bmat validation(sz, sz);

	mat a(sz, sz);
	mat b(sz, sz);

	a.randomize(0, 10);
	b.randomize(0, 10);

	BC_TEST_DEF(
		mat c(a);
		validation = (c + b).approx_equal(a += b);
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c - b).approx_equal(a -= b);
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c / b).approx_equal(a /= b);
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c % b).approx_equal(a %= b);
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c + 1) >= a;
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c - 1) <= a;
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = c == a;
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = c*-1 == -a;
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		c += 2;
		validation = c == a + 2;
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		c -= 2;
		validation = c == a - 2;
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		c /= 2;
		validation = c == a / 2;
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		c %= 2; //%= element-wise multiplication
		validation = c == a * 2; //scalar multiplication
		return BC::tensors::all(validation);
	)

	BC_TEST_BODY_TAIL
}

template<class value_type, template<class> class allocator=BC::Basic_Allocator>
int test_matrix_muls(int sz=128) {

	sz =16;
	BC_TEST_BODY_HEAD

	using alloc_t = allocator<value_type>;

	using mat = BC::Matrix<value_type, alloc_t>;
	using scal = BC::Scalar<value_type, alloc_t>;
	using bmat = BC::Matrix<bool, allocator<bool>>;

	bmat validation(sz, sz);

	mat a(sz, sz);
	mat b(sz, sz);
	mat c(sz, sz);
	mat d(sz, sz);

	scal alpha2;
	alpha2 = 2.0;

	a.randomize(0, 12);
	b.randomize(0, 12);


	validation.get_stream().get_allocator().force_deallocate();

	//lv trans
	BC_TEST_DEF(
		mat atrans = a.t();
		c = atrans.t() * b;
		d = a * b;

		validation =  c.approx_equal(d);
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat atrans = a.t();
		c = atrans.t() * b;
		d = a * b;
		validation =  c.approx_equal(d);
		return BC::tensors::all(validation);
	)

	//rv trans
	BC_TEST_DEF(
		mat atrans = a.t();
		c=(b * a);
		d=(b * atrans.t());
		validation = c.approx_equal(d);
		return BC::tensors::all(validation);
	)



//	BC_TEST_DEF(
//		scal two(2.f);
//
//		c = a * b * two;
//		d = two * a * b;
//
//		validation = c.approx_equal(d);
//		return BC::tensors::all(validation);
//	)


	BC_TEST_DEF(
		a.print();
		b.print();

		c = a * b * 2;
		d = 2 * a * b;

		validation = c.approx_equal(d);
		return BC::tensors::all(validation);
	)


	BC_TEST_DEF(
		c.zero();
		d.zero();

		c = a.t() * b * 2;
		d = 2 * a.t() * b;

		validation = c.approx_equal(d);
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat atrans(a.t());

		c = atrans * b * 2;
		d = 2 * a.t() * b;

		validation = c.approx_equal(d);
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat atrans(a.t());
		c = atrans * b * 2;
		d = 2 * atrans * b;

		validation = c.approx_equal(d);
		return BC::tensors::all(validation);
	)
	BC_TEST_DEF(
		mat atrans(a.t());
		c = atrans * b * 2.0f + 8.0f;
		d = 3 + 2 * atrans * b + 5;

		validation = c.approx_equal(d);
		return BC::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat atrans(a.t());
		c = (atrans * b * 2.0f + 8.0f) + (atrans * b * 2.0f + 8.0f);

		d = (3 + 2 * atrans * b + 5) * 2;
		mat e(sz, sz);
		e = (3 + 2 * atrans * b + 5);
		e += 2 * atrans * b;
		e += 8;

		mat f(sz, sz);
		f = (3 + 2 * atrans * b + 5);
		f += f;

		validation = (c.approx_equal(d) && c.approx_equal(e) && c.approx_equal(f));
		return BC::tensors::all(validation);
	)
	BC_TEST_DEF(
		mat atrans(a.t());

		c = (atrans * b * 2.0f - 8.0f) - (atrans * b * 2.0f - 8.0f);
		d = (3 - 2 * atrans * b - 5) - (3 - 2 * atrans * b - 5);

		mat e(sz, sz);
		e = (3 - 2 * atrans * b - 5);
		e -= (-2 * atrans * b) - 2;

		mat f(sz, sz);
		f = (3 - 2 * atrans * b - 5);
		f -= f;

		mat g(sz, sz);
		g =  (atrans * 5 * b - 5);
		g -= (atrans * 5 * b - 5);

		mat h = (a.t() * b * 2.0f - 8.0f) - (atrans * b * 2.0f - 8.0f);


		validation = c.approx_equal(d) && e.approx_equal(f) && g.approx_equal(h);
		return BC::tensors::all(validation);
	)
	validation.get_stream().get_allocator().force_deallocate();


	BC_TEST_DEF(
		mat y(4,4);
		mat dy(4,4);
		mat w(4,4);
		mat x(4,4);
		mat z(4,4);

		z += w.t() * BC::logistic.dx(w.t() * (y-dy));

		return z.get_stream().get_allocator().allocated_bytes() == 0;
	)

	BC_TEST_DEF(
		mat y(4,4);
		mat dy(4,4);
		mat w(4,4);
		mat x(4,4);
		mat z(4,4);

		z -= w.t() * BC::logistic.dx(w.t() * (y-dy));

		return z.get_stream().get_allocator().allocated_bytes() == 0;
	)

	BC_TEST_BODY_TAIL
}


}
}

#endif /* BLAS_Function_TESTING_H_ */
