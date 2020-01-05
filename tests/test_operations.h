/*
 * blas_function_testing.h
 *
 *  Created on: Dec 1, 2018
 *      Author: joseph
 */

#ifndef BC_BLAS_Function_TESTING_H_
#define BC_BLAS_Function_TESTING_H_

#include "test_common.h"

namespace bc {
namespace tests {

template<class value_type, template<class> class allocator=bc::Basic_Allocator>
int test_operations(int sz=128) {

	using alloc_t = allocator<value_type>;
	using mat =  bc::Matrix<value_type, alloc_t>;
	using bmat = bc::Matrix<bool, allocator<bool>>;

	BC_TEST_BODY_HEAD

	bmat validation(sz, sz);

	mat a(sz, sz);
	mat b(sz, sz);


	BC_TEST_ON_STARTUP {
		a.randomize(0, 10);
		b.randomize(0, 10);
	};

	BC_TEST_DEF(
		mat c(a);
		validation = (c + b).approx_equal(a += b);
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c - b).approx_equal(a -= b);
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c / b).approx_equal(a /= b);
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c % b).approx_equal(a %= b);
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c + 1) >= a;
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = (c - 1) <= a;
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = c == a;
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		validation = c*-1 == -a;
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		c += 2;
		validation = c == a + 2;
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		c -= 2;
		validation = c == a - 2;
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		c /= 2;
		validation = c == a / 2;
		return bc::tensors::all(validation);
	)

	BC_TEST_DEF(
		mat c(a);
		c %= 2; //%= element-wise multiplication
		validation = c == a * 2; //scalar multiplication
		return bc::tensors::all(validation);
	)

	BC_TEST_BODY_TAIL
}

template<class value_type, template<class> class allocator=bc::Basic_Allocator>
int test_matrix_muls(int sz=128) {

	BC_TEST_BODY_HEAD

	using alloc_t = allocator<value_type>;

	using mat = bc::Matrix<value_type, alloc_t>;
	using scal = bc::Scalar<value_type, alloc_t>;
	using bmat = bc::Matrix<bool, allocator<bool>>;

	bmat validation(sz, sz);

	mat a(sz, sz);
	mat b(sz, sz);
	mat c(sz, sz);
	mat d(sz, sz);

	scal alpha2;
	alpha2 = 2.0;

	auto default_stream_allocator_no_reserve = [&]() {
		return validation.get_stream().get_allocator().reserved_bytes() == 0;
	};

	auto default_stream_allocator_no_allocated= [&]() {
		return validation.get_stream().get_allocator().allocated_bytes() == 0;
	};

	auto no_allocation = [&]() {
		return default_stream_allocator_no_allocated() &&
				default_stream_allocator_no_reserve();
	};

	BC_TEST_ON_STARTUP {
		a.randomize(0, 12);
		b.randomize(0, 12);
		c.zero();
		d.zero();

		validation.get_stream().get_allocator().force_deallocate();
		BC_ASSERT(default_stream_allocator_no_allocated(),
				"Force deallocation expects no memory should be allocated");

		validation.get_stream().get_allocator().free();
		BC_ASSERT(default_stream_allocator_no_reserve(),
						"Free expects no memory should be reserved");
	};


	//lv trans
	BC_TEST_DEF(
		mat atrans = a.t();
		c = atrans.t() * b;
		d = a * b;

		validation =  c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	BC_TEST_DEF(
		mat atrans = a.t();
		c = atrans.t() * b;
		d = a * b;
		validation =  c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	//rv trans
	BC_TEST_DEF(
		mat atrans = a.t();
		c=(b * a);
		d=(b * atrans.t());
		validation = c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	BC_TEST_DEF(
		scal two(2.f);

		c = a * b * two;
		d = two * a * b;

		validation = c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	BC_TEST_DEF(
		c = a * b * 2;
		d = 2 * a * b;

		validation = c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	BC_TEST_DEF(
		c.zero();
		d.zero();

		c = a.t() * b * 2;
		d = 2 * a.t() * b;

		validation = c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	BC_TEST_DEF(
		mat atrans(a.t());

		c = atrans * b * 2;
		d = 2 * a.t() * b;

		validation = c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	BC_TEST_DEF(
		mat atrans(a.t());
		c = atrans * b * 2;
		d = 2 * atrans * b;

		validation = c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	BC_TEST_DEF(
		mat atrans(a.t());
		c = atrans * b * 2.0f + 8.0f;
		d = 3 + 2 * atrans * b + 5;

		validation = c.approx_equal(d);
		return bc::tensors::all(validation) && no_allocation();
	)

	BC_TEST_DEF(
		mat atrans(a.t());
		c = (atrans * b * 2.0f + 8.0f) + (atrans * b * 2.0f + 8.0f);
		bool c_no_alloc = no_allocation();

		d = (3 + 2 * atrans * b + 5) * 2;
		mat e(sz, sz);
		e = (3 + 2 * atrans * b + 5);
		e += 2 * atrans * b;
		e += 8;

		mat f(sz, sz);
		f = (3 + 2 * atrans * b + 5);
		f += f;

		validation = c.approx_equal(d) &&
				c.approx_equal(e) &&
				c.approx_equal(f);

		return bc::tensors::all(validation) && c_no_alloc;
	)

	BC_TEST_DEF(
		mat atrans(a.t());

		c = (atrans * b * 2.0f - 8.0f) - (atrans * b * 2.0f - 8.0f);
		bool c_no_alloc = no_allocation();

		d = (3 - 2 * atrans * b - 5) - (3 - 2 * atrans * b - 5);
		bool d_no_alloc = no_allocation();

		mat e(sz, sz);
		e = (3 - 2 * atrans * b - 5);
		e -= (-2 * atrans * b) - 2;
		bool e_no_alloc = no_allocation();

		mat f(sz, sz);
		f = (3 - 2 * atrans * b - 5);
		f -= f;
		bool f_no_alloc = no_allocation();


		mat g(sz, sz);
		g =  (atrans * 5 * b - 5);
		g -= (atrans * 5 * b - 5);
		bool g_no_alloc = no_allocation();

		mat h = (a.t() * b * 2.0f - 8.0f) - (atrans * b * 2.0f - 8.0f);

		validation = c.approx_equal(d) &&
				e.approx_equal(f) &&
				g.approx_equal(h);

		bool no_alloc =
				c_no_alloc &&
				d_no_alloc &&
				e_no_alloc &&
				f_no_alloc &&
				g_no_alloc;

		return bc::tensors::all(validation) && no_alloc;
	)

	BC_TEST_DEF(
		mat y(4,4);
		mat dy(4,4);
		mat w(4,4);
		mat x(4,4);
		mat z(4,4);

		z += w.t() * bc::logistic.dx(w.t() * (y-dy));
		return z.get_stream().get_allocator().allocated_bytes() == 0;
	)

	BC_TEST_DEF(
		mat y(4,4);
		mat dy(4,4);
		mat w(4,4);
		mat x(4,4);
		mat z(4,4);

		z -= w.t() * bc::logistic.dx(w.t() * (y-dy));
		return z.get_stream().get_allocator().allocated_bytes() == 0;
	)

	BC_TEST_BODY_TAIL
}


}
}

#endif /* BLAS_Function_TESTING_H_ */
