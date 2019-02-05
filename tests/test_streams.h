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
	using vec = BC::Vector<value_type, log_allocator<allocator<value_type>>>;



	//sanity check
	BC_TEST_DEF(
			mat a;
			mat b;
		return a.is_default_stream() && b.is_default_stream();
	)

	BC_TEST_DEF(
		mat a;
		mat b;

		a.create_stream();
		b.create_stream();
		bool new_stream =  !a.is_default_stream() && !b.is_default_stream();

		a.destroy_stream();
		b.destroy_stream();

		bool destroy_stream = a.is_default_stream() && b.is_default_stream();

		return new_stream && destroy_stream;
	)

	/*
	 * A full context is a struct that carries meta-data about the tensor.
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

		a.create_stream();

		return a.get_full_context() != b.get_full_context();
	)

	BC_TEST_DEF(
		mat a;
		mat b;

		a.create_stream();
		b.set_stream(a);

		return a.get_full_context() == b.get_full_context() && !a.is_default_stream() && !b.is_default_stream();
	)

	BC_TEST_DEF(
		mat a(4,4);
		a.create_stream();

	    auto col_0 = a[0];
	    auto scal_0 = col_0[0];

		return !a.is_default_stream() &&
				a.get_full_context() == col_0.get_full_context() &&
				a.get_full_context() == scal_0.get_full_context();
	)

	BC_TEST_DEF(
		mat a(4,4);
		a.create_stream();

	    auto col_0 = a[0];
	    auto scal_0 = col_0[0];

	    a.destroy_stream();

		return a.is_default_stream() &&
				a.get_full_context() != col_0.get_full_context() &&
				a.get_full_context() != scal_0.get_full_context() &&
				col_0.get_full_context() == scal_0.get_full_context();
	)




	BC_TEST_BODY_TAIL

}

}
}




#endif /* TEST_STREAMS_H_ */
