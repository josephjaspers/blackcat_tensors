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

namespace bc {
namespace tests {


template<class Allocator>
struct log_allocator : Allocator {

	using value_type = typename Allocator::value_type;

	std::shared_ptr<bc::size_t> total_allocated =
			std::shared_ptr<bc::size_t>(new bc::size_t());

	std::shared_ptr<bc::size_t> total_deallocated =
			std::shared_ptr<bc::size_t>(new bc::size_t());

	template<class T>
	struct rebind {
		using other = log_allocator<
				typename Allocator::template rebind<T>::other>;
	};

	log_allocator() = default;

	template<class T>
	log_allocator(const log_allocator<T>& la)
	{
		total_allocated = la.total_allocated;
		total_deallocated = la.total_deallocated;
	}

	auto allocate(bc::size_t sz)
	{
		(*total_allocated.get()) += sz * sizeof(value_type);
		return Allocator::allocate(sz);
	}

	auto deallocate(typename Allocator::value_type* data, bc::size_t sz)
	{
		(*total_deallocated) += sz * sizeof(value_type);
		return Allocator::deallocate(data, sz);

	}
};


template<class value_type, template<class> class allocator>
int test_allocators(int sz=128) {

	using bc::tensors::all;

	BC_TEST_BODY_HEAD

	using allocator_type = allocator<value_type>;
	using mat = bc::Matrix<value_type, log_allocator<allocator_type>>;
	using vec = bc::Vector<value_type, log_allocator<allocator_type>>;
	using system_tag = typename allocator_traits<allocator_type>::system_tag;

	Stream<system_tag> stream;

	BC_TEST_ON_STARTUP
	{
		stream.get_allocator().free();
		stream.get_allocator().set_allocator(log_allocator<allocator_type>());
		mat tmp;
		tmp.get_stream().get_allocator().set_allocator(log_allocator<allocator_type>());
	};

	BC_TEST_DEF(
		mat a(5,5);
		return *(a.get_allocator().total_allocated.get())
				== 25 * sizeof(value_type);
	)

	BC_TEST_DEF(
		mat a(5,5);
		mat b(a);
		return *(b.get_allocator().total_allocated.get())
				== 50 * sizeof(value_type);
	)

	BC_TEST_DEF(
		mat a(5,5);  //mem sz = 25
		vec b(a[0]); //       = 30 (allocators should propagate from slices)

		return *(b.get_allocator().total_allocated)
				== 30 * sizeof(value_type);
	)

	BC_TEST_DEF(
		mat a(5,5);  //mem sz = 25
		a.get_stream().get_allocator().reserve(30 * sizeof(value_type));
		a = bc::logistic(a * a + a); // should not allocate any memory
		return *(a.get_allocator().total_allocated.get()) == 25 * sizeof(value_type);
	)

	BC_TEST_BODY_TAIL
};





}
}



#endif /* TEST_ALLOCATORS_H_ */
