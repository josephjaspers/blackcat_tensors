#include "test_accessors.h"
#include "test_allocators.h"
#include "test_common.h"
#include "test_constructors.h"
#include "test_operations.h"
#include "test_streams.h"
#include "test_algorithms.h"
#include "test_blas.h"
#include "test_neuralnetworks.h"

namespace bc {
namespace tests {

template<class scalar_type, template<class> class allocator>
int test_all(int sz) {

	using namespace bc::tests;

	int errors = 0;
	errors += test_accessors<scalar_type, allocator>(sz);
	errors += test_allocators<scalar_type, allocator>(sz);
	errors += test_constructors<scalar_type, allocator>(sz);
	errors += test_operations<scalar_type, allocator>(sz);
	errors += test_matrix_muls<scalar_type, allocator>(sz);
	errors += test_streams<scalar_type, allocator>(sz);
	errors += test_algorithms<scalar_type, allocator>(sz);
	errors += test_blas<scalar_type, allocator>(sz);
	errors += test_neuralnetworks<scalar_type, allocator>(sz);
	return errors;
}


int run(int sz=64) {

	int errors = 0;

	errors += test_all<double, bc::Basic_Allocator>(sz);
	errors += test_all<double, std::allocator>(sz);

#ifdef __CUDACC__
	errors += test_all<float, bc::Cuda_Allocator>(sz);
#endif

	if (!errors) {
		std::cout << "All Tests Successful" << std::endl;
	} else {
		std::cout << "BC Tests failure: " << errors << " tests failed" << std::endl;
	}
	return errors;
}


}
}
