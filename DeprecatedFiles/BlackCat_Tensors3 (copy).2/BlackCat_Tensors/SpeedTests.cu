//#include <iostream>
//#include <omp.h>
//#include "math.h"
//#include "time.h"
//
//#include "BlackCat_Tensors.h"
//
//
//using BC::Vector;
//
//template<int SIZE, int repetitions>
//int speedTests() {
//	const int reps = repetitions;
//
//	std::cout << "  ------------------------------------------ speed tests ------------------------------------------" << std::endl;
//	std::cout << " size = " << SIZE << std::endl;
//	std::cout << " Repetitions = " << reps << std::endl;
//
//	using vec = Vector<double>;
////	using vec = Vector<float, SIZE, BC::GPU>;
//	vec a(SIZE);
//	vec b(SIZE);
//	vec c(SIZE);
//	vec d(SIZE);
//
//	b.randomize(0, 10);
//	c.randomize(0, 10);
//	b.randomize(0, 10);
//
//	float t;
//
//	t = omp_get_wtime();
//	printf("\n Calculating... (BlackCat_Tensors) a = b + c - d % b + c - d; \n");
//	for (int i = 0; i < reps; ++i) {
//		//a = b + c - d % b + c - d;
//		a = b + c + d;
//	}
//
//
//
////	at.print();
//
//	t = omp_get_wtime() - t;
//	printf("It took me %f clicks (%f seconds).\n", t, ((float) t));
//
//	if (SIZE < 100) {
//		b.print();
//		c.print();
//		a.print();
//	}
//	std::cout << "success " << std::endl;
//
//#pragma omp barrier
//	return 0;
//}
//int main() {
//
//	speedTests<100000, 10000>();
//	speedTests<10000, 100000>();
//	speedTests<1000, 1000000>();
//	speedTests<100, 1000000>();
//	speedTests<10, 1000000>();
//
//	return 0;
//}
//
//
