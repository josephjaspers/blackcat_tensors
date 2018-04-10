#include "MNIST_test.h"
#include "MNIST_test_MT.h"

#define BC_DISABLE_OPENMP

int main() {

//	BC::MNIST_Test::percept_MNIST();
	BC::MNIST_Test_MT::percept_MNIST();
	std::cout << "terminate success "<< std::endl;


	using BC::vec;
	BC::bp_list<vec&> list(8);
//	BC::Structure::forward_list<vec> list;

	vec a(10);
	vec b(a);

	b.print();
	a.print();
	std::cout << "pushing " << std::endl;
	list().push_front(a);
//	a.print();

//	list().front().print();
	std::cout << "end" << std::endl;
}
