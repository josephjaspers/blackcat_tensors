///*
// * main.cpp
// *
// *  Created on: Dec 1, 2017
// *      Author: joseph
// */
//
//#include "BC_Expression_Binary_Pointwise_Same.h"
//#include "BC_Internals_Include.h"
//#include "BC_Tensor_BaseClass_Lv1_Vector.h"
//#include "BC_Tensor_InheritLv1_FunctionType.h"
//#include "BC_Tensor_InheritLv1_Shape.h"
//#include "BC_Tensor_InheritLv3_King.h"
//#include "BC_Tensor_BaseClass_Lv2_Matrix.h"
//#include "BC_Tensor_InheritLv5_Core.h"
//
//int main2() {
//
//	Vector<double, CPU, 10> alpha;
//	alpha.print();
//
//	Vector<double, CPU, 10> beta;
//	beta.randomize(-3, 3);
//	beta[0].print();
//
//	alpha.randomize(-3, 3);
//	alpha.print();
//
//	alpha = alpha + alpha;
//
//	alpha.print();
//
//	beta[0].print();
//
//	alpha = alpha + beta[0];
//	alpha.print();
//
//	alpha.t().print();
//
//	std::cout<< " rows cols       = " << alpha.rows() << " " << alpha.cols() << std::endl;
//	std::cout<< " rows cols trans = " << alpha.t().rows() << " " << alpha.t().cols() << std::endl;
//
//
//	Matrix<double, CPU, 5, 7> mat;
//
//	mat.randomize(03, 5);
//	mat.print();
//
//	Matrix<double, CPU, 7, 5> matT;
//	matT = mat.t();
//	matT.print();
//
//	std::cout << " success " << std::endl;
//	return 0;
//}
//
