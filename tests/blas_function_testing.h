/*
 * blas_function_testing.h
 *
 *  Created on: Dec 1, 2018
 *      Author: joseph
 */

#ifndef BLAS_FUNCTION_TESTING_H_
#define BLAS_FUNCTION_TESTING_H_

#include "../include/BlackCat_Tensors.h"

namespace BC {
namespace tests {

template<class scalar_t, class alloc_t=BC::Basic_Allocator>
int test_operations(int sz=4) {

	using mat = BC::Matrix<scalar_t, alloc_t>;
	using vec = BC::Vector<scalar_t, alloc_t>;
	using bmat = BC::Matrix<bool, alloc_t>;

	int errors = 0;

	bmat validation(sz, sz);

	mat a(sz, sz);
	mat b(sz, sz);

	a.randomize(0, 10);
	b.randomize(0, 10);

	{
		mat c(a);
		validation = (c + b).approx_equal(a += b);
		errors += !BC::all(validation);
	}
	{
		mat c(a);
		validation = (c - b).approx_equal(a -= b);
		errors += !BC::all(validation);
	}
	{
		mat c(a);
		validation = (c / b).approx_equal(a /= b);
		errors += !BC::all(validation);
	}
	{
		mat c(a);
		validation = (c % b).approx_equal(a %= b);
		errors += !BC::all(validation);
	}
	{
		mat c(a);
		validation = (c + 1) >= a;
		errors += !BC::all(validation);
	}
	{
		mat c(a);
		validation = (c - 1) <= a;
		errors += !BC::all(validation);
	}
	{
		mat c(a);
		validation = c == a;
		errors += !BC::all(validation);
	}
	{
		mat c(a);
		validation = c*-1 == -a;
		errors += !BC::all(validation);
	}

	return errors;
}
template<class scalar_t, class alloc_t=BC::Basic_Allocator>
int test_matrix_muls(int sz=4) {

	using mat = BC::Matrix<scalar_t, alloc_t>;
	using vec = BC::Vector<scalar_t, alloc_t>;
	using scal = BC::Scalar<scalar_t, alloc_t>;
	using bmat = BC::Matrix<bool, alloc_t>;

	int errors = 0;

	bmat validation(sz, sz);

	mat a(sz, sz);
	mat b(sz, sz);
	mat c(sz, sz);
	mat d(sz, sz);

	scal alpha2;
	alpha2 = 2.0;

	a.randomize(0, 10);
	b.randomize(0, 10);


#define BC_TEST_MATMUL(code)\
	{\
		auto test = [&]() { code };\
		if (!test()) {\
			std::cout << "TEST FAILURE: " #code  << std::endl;\
			errors++; \
		}\
	}

	//lv trans
	BC_TEST_MATMUL(
		mat atrans = a.t();
		c = atrans.t() * b;
		d = a * b;
		validation =  c.approx_equal(d);
		return BC::all(validation);
	)

	BC_TEST_MATMUL(
		mat atrans = a.t();
		c = atrans.t() * b;
		d = a * b;
		validation =  c.approx_equal(d);
		return BC::all(validation);
	)

	//rv trans
	BC_TEST_MATMUL(
		mat atrans = a.t();
		c=(b * a);
		d=(b * atrans.t());
		validation = c.approx_equal(d);
		return BC::all(validation);
	)

	BC_TEST_MATMUL(
		c = a * b * 2;
		d = 2 * a * b;

		validation = c.approx_equal(d);
		return BC::all(validation);
	)


	BC_TEST_MATMUL(
		c = a.t() * b * 2;
		d = 2 * a.t() * b;

		validation = c.approx_equal(d);
		return BC::all(validation);
	)
	BC_TEST_MATMUL(
		mat atrans(a.t());
		c = atrans * b * 2;
		d = 2 * a.t() * b;

		validation = c.approx_equal(d);
		return BC::all(validation);
	)
	BC_TEST_MATMUL(
		mat atrans(a.t());
		c = atrans * b * 2;
		d = 2 * atrans * b;

		validation = c.approx_equal(d);
		return BC::all(validation);
	)
	BC_TEST_MATMUL(
		mat atrans(a.t());
		c = atrans * b * 2 + 8;
//		d = 3 + 2 * atrans * b + 5;

		validation = c.approx_equal(d);
		return BC::all(validation);
	)

//	//lv trans
//	BC_TEST_MATMUL(
//		mat atrans = a.t();
//		validation = (atrans.t() * b * 2) == (b * a * 2);
//		return BC::all(validation);
//	)
//	//lv trans
//	BC_TEST_MATMUL(
//		mat atrans = a.t();
//		validation = (atrans.t() * 2 * b) == (b * a * 2);
//		return BC::all(validation);
//	)
//	//lv trans
//	BC_TEST_MATMUL(
//		mat atrans = a.t();
//		validation = (2 * atrans.t() * b) == (b * a * 2);
//		return BC::all(validation);
//	)
//	//lv trans
//	BC_TEST_MATMUL(
//		mat atrans = a.t();
//		validation = (atrans.t() * b * alpha2) == (b * a * 2);
//		return BC::all(validation);
//	)
//	//lv trans
//	BC_TEST_MATMUL(
//		mat atrans = a.t();
//		validation = (atrans.t() * alpha2 * b) == (b * a * 2);
//		return BC::all(validation);
//	)
//	//lv trans
//	BC_TEST_MATMUL(
//		mat atrans = a.t();
//		validation = (alpha2 * atrans.t() * b) == (b * a * 2);
//		return BC::all(validation);
//	)

	return errors;
}
}
}

#endif /* BLAS_FUNCTION_TESTING_H_ */
