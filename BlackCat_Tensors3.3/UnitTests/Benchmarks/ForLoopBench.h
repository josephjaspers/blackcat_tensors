///*
// * ForLoopBench.h
// *
// *  Created on: May 8, 2018
// *      Author: joseph
// */
//
//#ifndef FORLOOPBENCH_H_
//#define FORLOOPBENCH_H_
//
//
////#define BC_RELEASE;
//
//#include <omp.h>
////using vec = BC::Vector<double, BC::CPU>;
//
//int test2() {
//
//	const int size = 10000;
//	const int reps = 10000;
//	 vec a(size);
//	 vec b(size);
//	 vec c(size);
//	 vec d(size);
//
//	 double a_[size];
//	 double b_[size];
//	 double c_[size];
//	 double d_[size];
//
//	for (int i = 0; i < size; ++i) {
//		a_[i] = i;
//		b_[i] = i;
//		c_[i] = i;
//		d_[i] = i;
//		a(i) = i;
//		b(i) = i;
//		c(i) = i;
//		d(i) = i;
//	}
//
//	std::cout << " starting " << std::endl;
//	{
//	float t= omp_get_wtime();
//	for (int i = 0; i < reps; ++i) {
//
//		a += b + c - d;
//	}
//
//	t = omp_get_wtime() - t;
//	volatile float s = a(2);
//
//
//	  printf ("It took me %d clicks (%f seconds).\n",t,((float)t));
//	}
//a.print();
//
//	{
//	float t= omp_get_wtime();
//	for (int i = 0; i < reps; ++i) {
//		for (int j = 0; j < size; ++j)
//		a_[j] += b_[j] + c_[j] - d_[j];
//	}
//
//
//	t = omp_get_wtime() - t;
//
//	volatile float s = a_[2];
//
//	  printf ("It took me %d clicks (%f seconds).\n",t,((float)t));
//	}
//}
//
//
//
//
//#endif /* FORLOOPBENCH_H_ */
