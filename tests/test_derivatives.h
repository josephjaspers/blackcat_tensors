/*
 * test_accessors.h
 *
 *  Created on: Dec 16, 2018
 *      Author: joseph
 */

#ifndef TEST_DERIVATIVES_H_
#define TEST_DERIVATIVES_H_

#include "test_common.h"

namespace BC {
namespace tests {

template<class value_type, template<class> class allocator>
int test_derivatives(int sz=128) {

	BC_TEST_BODY_HEAD

	using mat = BC::Matrix<value_type, allocator<value_type>>;

	using namespace BC::oper;
	using BC::tanh;
	using BC::cos;
	using BC::sin;
	using BC::tensors::dx;
	using BC::tensors::all;


	mat x(sz,sz);

	x = .5;

	//dx(sin(cos(x))) == cos(cos(x)) % -sin(x); // % == elementwise multiplication

	BC_TEST_DEF(
			mat out(sz,sz);
			out = dx(sin(cos(x)));
			return BC::all(out.approx_equal(-0.306359));
	)

	BC_TEST_DEF(
			mat out(sz,sz);
			return BC::all(
					dx(sin(cos(x))) == cos(cos(x)) % -sin(x)
				);
	)

	BC_TEST_DEF(
			mat out(sz,sz);
			out = cos(cos(x)) % -sin(x);
			return BC::all(out.approx_equal(-0.306359));
	)



	BC_TEST_DEF(
			mat out(sz,sz);
			out = dx(sin(cos(x)) + sin(cos(x)));
			return BC::all(out.approx_equal(-0.306359 * 2));
	)

	BC_TEST_DEF(
			mat out(sz,sz);
			out = dx(sin(cos(x)) % sin(cos(x)));
			return BC::all(out.approx_equal(-0.471300));
	)

	BC_TEST_DEF(
			mat out(sz,sz);
			out = dx(sin(cos(x) + cos(x)));
			return BC::all(out.approx_equal(0.175782));
	)

	BC_TEST_BODY_TAIL
}
}
}



#endif /* TEST_ACCESSORS_H_ */
