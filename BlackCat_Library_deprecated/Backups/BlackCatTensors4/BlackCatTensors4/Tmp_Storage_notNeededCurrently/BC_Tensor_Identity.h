///*
// * BC_Tensor_Super_Identity.h
// *
// *  Created on: Nov 20, 2017
// *      Author: joseph
// */
//
//#ifndef BC_TENSOR_IDENTITY_H_
//#define BC_TENSOR_IDENTITY_H_
//
//struct Scalar;
//struct Vector;
//struct Matrix;
//struct Cube;
//struct Tensor;
//
//struct row_Vector;
//struct sub_Vector;
//struct sub_Matrix;
//struct sub_Cube;
//struct sub_Tensor;
//
//template<int... dims>
//struct Shape;
//
//template<class dimensions, class leading_dimensions = void>
//struct Identity {
//	using tensor_type = void;
//};
//
//template<>
//struct Identity<Shape<1>> {
//	using tensor_type = Scalar;
//};
//
//template<int dim>
//struct Identity<Shape<dim>> {
//	using tensor_type = Vector;
//};
//
//template<int row, int col>
//struct Identity<Shape<row, col>> {
//	using tensor_type = Matrix;
//};
//
//template<int row, int col, int depth>
//struct Identity<Shape<row, col, depth>> {
//	using tensor_type = Cube;
//};
//
//template<int... dims>
//struct Identity<Shape<dims...>> {
//	using tensor_type = Tensor;
//};
//
//template<int dim, int ld>
//struct Identity<Shape<dim>, Shape<ld>> {
//	using tensor_type = row_Vector;
//};
//
//template<int row, int col, int lead_row_height>
//struct Identity<Shape<row, col>, Shape<lead_row_height>> {
//	using tensor_type = sub_Matrix;
//};
//
//template<int row, int col, int depth, int ld1, int ld2>
//struct Identity<Shape<row, col, depth>, Shape<ld1, ld2>> {
//	using tensor_type = sub_Cube;
//};
//
//template<int... dims, int... ld_dims>
//struct Identity<Shape<dims...>, Shape<ld_dims...>> {
//	using tensor_type = sub_Tensor;
//};
//
//
//
//#endif /* BC_TENSOR_IDENTITY_H_ */
