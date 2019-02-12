/*
 * test_algorithms.h
 *
 *  Created on: Feb 6, 2019
 *      Author: joseph
 */

#ifndef BC_TESTS_ALGORITHMS_H_
#define BC_TESTS_ALGORITHMS_H_

#include "test_common.h"

namespace BC {
namespace tests {

template<class value_type, template<class> class allocator>
int test_algorithms(int sz=128) {

	BC_TEST_BODY_HEAD

	using alloc_t = allocator<value_type>;
	using vec = BC::Vector<value_type, alloc_t>;
	using mat = BC::Matrix<value_type, alloc_t>;
	using scalar = BC::Scalar<value_type, alloc_t>;

	BC_TEST_DEF(
		mat a(sz, sz);
		a.fill(2);

		return BC::all(a == 2);
	)
	BC_TEST_DEF(
		mat a(sz, sz);
		BC::fill(a.begin(), a.end(), 3);

		return BC::all(a == 3);
	)

	BC_TEST_DEF(
		mat a(sz, sz);
		mat b(sz, sz);

		a.create_stream();
		b.create_stream();

		BC::fill(a.get_stream(), a.begin(), a.end(), 5);
		BC::fill(b.get_stream(), b.begin(), b.end(), 7);

		a.sync_stream();
		b.sync_stream();


		return BC::all(a == 5) && BC::all(b == 7);
	)


	//same test as above except B is using the default stream
	BC_TEST_DEF(
		mat a(sz, sz);
		mat b(sz, sz);

		a.create_stream();

		BC::fill(a.get_stream(), a.begin(), a.end(), 5);
		BC::fill(b.get_stream(), b.begin(), b.end(), 7);

		a.sync_stream();
		b.sync_stream();


		return BC::all(a == 5) && BC::all(b == 7);
	)

	BC_TEST_BODY_TAIL
}





}
}





#endif /* TEST_ALGORITHMS_H_ */
