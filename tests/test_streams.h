/*
 * test_streams.h
 *
 *  Created on: Feb 4, 2019
 *      Author: joseph
 */

#ifndef TEST_STREAMS_H_
#define TEST_STREAMS_H_

#include "test_common.h"

namespace BC {
namespace tests {


template<class value_type, template<class> class allocator>
int test_streams(int sz=128) {

	BC_TEST_BODY_HEAD

	using mat = BC::Matrix<value_type, log_allocator<allocator<value_type>>>;



	//sanity check
	BC_TEST_DEF(
			mat a;
			mat b;
		return a.get_stream().is_default() && b.get_stream().is_default();
	)

	BC_TEST_DEF(
		mat a;
		mat b;

		a.get_stream().create();
		b.get_stream().create();
		bool new_stream =  !a.get_stream().is_default() && !b.get_stream().is_default();

		a.get_stream().destroy();
		b.get_stream().destroy();

		bool destroy = a.get_stream().is_default() && b.get_stream().is_default();

		return new_stream && destroy;
	)

	/*
	 * A full stream is a struct that carries meta-data about the tensor.
	 * Depending if it is a GPU/CPU tensor it stores....
	 * 	cublas-handle,
	 * 	allocator,
	 * 	stream,
	 * 	(small) memory buffers
	 *
	 */

	BC_TEST_DEF(
		mat a;
		mat b;

		a.get_stream().create();

		return a.get_stream() != b.get_stream();
	)

	BC_TEST_DEF(
		mat a;
		mat b;

		a.get_stream().create();
		b.get_stream().set_stream(a.get_stream());

		return a.get_stream() == b.get_stream() && !a.get_stream().is_default() && !b.get_stream().is_default();
	)

	BC_TEST_DEF(
		mat a(4,4);
		a.get_stream().create();

	    auto col_0 = a[0];
	    auto scal_0 = col_0[0];

		return !a.get_stream().is_default() &&
				a.get_stream() == col_0.get_stream() &&
				a.get_stream() == scal_0.get_stream();
	)

	BC_TEST_DEF(
		mat a(4,4);
		a.get_stream().create();

	    auto col_0 = a[0];
	    auto scal_0 = col_0[0];

	    a.get_stream().destroy();

		return a.get_stream().is_default() &&
				a.get_stream() != col_0.get_stream() &&
				a.get_stream() != scal_0.get_stream() &&
				col_0.get_stream() == scal_0.get_stream();
	)

	BC_TEST_BODY_TAIL
}

}
}




#endif /* TEST_STREAMS_H_ */
