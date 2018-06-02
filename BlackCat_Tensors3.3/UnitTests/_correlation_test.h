///*
// * _correlation_test.h
// *
// *  Created on: May 22, 2018
// *      Author: joseph
// */
//
//#ifndef CORRELATION_TEST_H_
//#define CORRELATION_TEST_H_
//
//#include <iostream>
//#include "../BlackCat_Tensors.h"
//
//using BC::Vector;
//using BC::Matrix;
//
//int correlation() {
//	std::cout << " --------------------------------------CORRELATION--------------------------------------" << std::endl;
//
//
//	mat krnl(2,2);
//	krnl.zero();
//
//	krnl[0][0] = 1;
//	krnl[1][1] = 1;
//
//
//	mat img(5,5);
//	img.zero();
//
//	for (int m = 0; m < img.rows(); ++m) {
//		for (int n = 0; n < img.cols(); ++n)
//			if (m ==n)
//				img[m][n] = 1;
//	}
//
//
//	std::cout << "kernel is ..." << std::endl;
//	krnl.print();
//	std::cout << "image is ..." << std::endl;
//	img.print();
//
//	std::cout << "output is " << std::endl;
//	m.print();
//
//	std::cout << " 2-d x_corr (3d tensor)--------------" << std::endl;
//	cube krnl2(2,2,3);
//	cube img2(8,8,3);
//
//
//	for (int m = 0; m < img2.dimension(2); ++m)
//		for (int n = 0; n < img2.dimension(1); ++n) {
//			for (int k = 0; k < img2.dimension(0); ++k) {
//				if (n == k)
//				img2[m][n][k] = 1;
//			}
//		}
//	for (int m = 0; m < krnl2.dimension(2); ++m)
//		for (int n = 0; n < krnl2.dimension(1); ++n) {
//			for (int k = 0; k < krnl2.dimension(0); ++k) {
//				if (n == k)
//					krnl2[m][n][k] = 1;
//			}
//		}
//	std::cout << "kernel is ..." << std::endl;
//	krnl2.print();
//	std::cout << "image is ..." << std::endl;
//	img2.print();
//
//
//	mat out2 = krnl2.x_corr<2>(img2);
//
//	std::cout << "output is " << std::endl;
//	out2.print();
//
//
//
//	return 0;
//}
//
//
//#endif /* CORRELATION_TEST_H_ */
