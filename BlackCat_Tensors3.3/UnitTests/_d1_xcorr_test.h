/*
 * _d1_xcorr_test.h
 *
 *  Created on: May 22, 2018
 *      Author: joseph
 */

#ifndef D1_XCORR_TEST_H_
#define D1_XCORR_TEST_H_


#include <iostream>
#include "../BlackCat_Tensors.h"
#include <omp.h>

using BC::Vector;
using BC::Matrix;

int imgdim(int length, int krnl_length, int depth) {
	return length - krnl_length * depth + depth;
}

template<class ml = BC::CPU>
int d1_xcorr(int KL, int IL, int iterations = 100) {
	using tensor = vec;

	tensor k1(KL); k1.randomize(0, 10);
	tensor k2(KL); k2.randomize(0, 10);
	tensor k3(KL); k3.randomize(0, 10);

	tensor img1((imgdim(IL, KL, 0))); img1.randomize(0, 10);
	tensor img2((imgdim(IL, KL, 1))); img2.randomize(0, 10);
	tensor img3((imgdim(IL, KL, 2))); img3.randomize(0, 10);
	tensor img4((imgdim(IL, KL, 3))); img4.randomize(0, 10);


		float seg = omp_get_wtime();
	for (int i = 0; i < iterations; ++i) {
		img2 = k1.x_corr<1>(img1);
		img3 = k2.x_corr<1>(img2);
		img4 = k3.x_corr<1>(img3);
	}
	seg = omp_get_wtime() - seg;


	float chain = omp_get_wtime();
	for (int i = 0; i < iterations; ++i) {
		img4 = k3.x_corr<1>(img3 =* k2.x_corr<1>(img2 =* k1.x_corr<1>(img1)));
	}
	chain = omp_get_wtime() - chain;


	std::cout << "img length = " << IL << "krnl length" << KL <<  std::endl<< "  segmented time = " << seg << " || chained time = " << chain << std::endl;
	std::cout << "% difference - " << seg / chain << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

	return 0;
}

template<class ml = BC::CPU>
int d1_xcorr_mat(int KL, int IL, int iterations = 1) {
	using tensor = mat;

	tensor k1(KL,KL); k1.randomize(0, 10);
	tensor k2(KL,KL); k2.randomize(0, 10);
	tensor k3(KL,KL); k3.randomize(0, 10);

	tensor img1(imgdim(IL, KL, 0),imgdim(IL, KL, 0)); img1.randomize(0, 10);
	tensor img2(imgdim(IL, KL, 1),imgdim(IL, KL, 1)); img2.randomize(0, 10);
	tensor img3(imgdim(IL, KL, 2),imgdim(IL, KL, 2)); img3.randomize(0, 10);
	tensor img4(imgdim(IL, KL, 3),imgdim(IL, KL, 3)); img4.randomize(0, 10);


		float seg = omp_get_wtime();
	for (int i = 0; i < iterations; ++i) {
		img2 = k1.x_corr<2>(img1);
		img3 = k2.x_corr<2>(img2);
		img4 = k3.x_corr<2>(img3);
	}
	seg = omp_get_wtime() - seg;


	float chain = omp_get_wtime();
	for (int i = 0; i < iterations; ++i) {
		img4 = k3.x_corr<2>(img3 =* k2.x_corr<2>(img2 =* k1.x_corr<2>(img1)));
	}
	chain = omp_get_wtime() - chain;


	std::cout << "img length = " << IL << "krnl length" << KL <<  std::endl<< "  segmented time = " << seg << " || chained time = " << chain << std::endl;
	std::cout << "% difference - " << seg / chain << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;

	return 0;
}



#endif /* D1_XCORR_TEST_H_ */
