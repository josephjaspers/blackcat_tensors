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

	using allocator_type = allocator<value_type>;
	using system_tag = typename BC::allocator_traits<allocator_type>::system_tag;
	using mat = BC::Matrix<value_type, allocator_type>;

	BC::streams::Stream<system_tag> stream;

	BC_TEST_DEF(
		mat a(sz, sz);
		a.fill(2);

		return BC::tensors::all(a == 2);
	)
	BC_TEST_DEF(
		mat a(sz, sz);
		BC::algorithms::fill(a.get_stream(), a.cw_begin(), a.cw_end(), 3);

		return BC::tensors::all(a == 3);
	)

	BC_TEST_DEF(
		mat a(sz, sz);
		mat b(sz, sz);

		a.get_stream().create();
		b.get_stream().create();

		BC::algorithms::fill(a.get_stream(), a.cw_begin(), a.cw_end(), 5);
		BC::algorithms::fill(b.get_stream(), b.cw_begin(), b.cw_end(), 7);

		a.get_stream().sync();
		b.get_stream().sync();

		return BC::tensors::all(a == 5) && BC::tensors::all(b == 7);
	)


	//same test as above except B is using the default stream
	BC_TEST_DEF(

		mat a(sz, sz);
		mat b(sz, sz);

		a.get_stream().create();

		BC::algorithms::fill(a.get_stream(), a.cw_begin(), a.cw_end(), 5);
		BC::algorithms::fill(b.get_stream(), b.cw_begin(), b.cw_end(), 7);

		a.get_stream().sync();
		b.get_stream().sync();


		return BC::tensors::all(a == 5) && BC::tensors::all(b == 7);
	)

	BC_TEST_BODY_TAIL
}





}
}





#endif /* TEST_ALGORITHMS_H_ */
