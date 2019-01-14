/*
 * test_allocators.h
 *
 *  Created on: Jan 13, 2019
 *      Author: joseph
 */

#ifndef TEST_ALLOCATORS_H_
#define TEST_ALLOCATORS_H_

#include "test_common.h"
#include <memory>

namespace BC {
namespace tests {

template<class value_type, template<class> class allocator>
int test_allocators(int sz=128) {

	BC_TEST_BODY_HEAD

	struct log_allocator : allocator<value_type> {

		using propagate_on_temporary_creation = std::true_type;

		std::shared_ptr<BC::size_t> total_mem = std::shared_ptr<BC::size_t>(new BC::size_t());
		std::shared_ptr<BC::size_t> mem_instances = std::shared_ptr<BC::size_t>(new BC::size_t());;

		auto allocate(BC::size_t sz) {
			(*total_mem.get()) += sz;
			(*mem_instances.get())++;
			return allocator<value_type>::allocate(sz);
		}
	};
	using mat = BC::Matrix<value_type, log_allocator>;
	using vec = BC::Vector<value_type, log_allocator>;

	BC_TEST_DEF(
		mat a(5,5);
		return *(a.get_allocator().total_mem.get()) == 25;
	)

	BC_TEST_DEF(
		mat a(5,5);
		mat b(a);
		return *(b.get_allocator().total_mem.get()) == 50;
	)


	//Allocators can be passed from slices
	BC_TEST_DEF(
		mat a(5,5);  //mem sz = 25
		vec b(a[0]); //       = 30 (allocators should propagate from slices)

		return *(b.get_allocator().total_mem.get()) == 30;
	)



	BC_TEST_BODY_TAIL


	};


}
}



#endif /* TEST_ALLOCATORS_H_ */
