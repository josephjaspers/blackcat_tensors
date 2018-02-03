///*
// * test_area.cu
// *
// *  Created on: Jan 11, 2018
// *      Author: joseph
// */
//
//#ifndef TEST_AREA_CU_
//#define TEST_AREA_CU_
//
//#include <omp.h>
//#include <stdio.h>
//
//
//int threads() {
//    return 256;
//}
//int blocks(int size) {
//    return (size + threads() - 1) / threads();
//}
//
//__global__
//void add_kernel(float* a, float* b, float* c, int size) {
//   for (int i = 0; i < size; ++i) {
//       a[i] = b[i] + c[i];
//   }
//}
//
//
//__global__
//void add_kernel(float* a, float* b, float* c, int size, int reps) {
//    for (int j = 0; j < reps; ++j)
//   for (int i = 0; i < size; ++i) {
//       a[i] = b[i] + c[i];
//   }
//}
//
//int main() {
//int sz = 1000; //Or any arbitrarily large number
//int reps = 1000;   //Or any arbitrarily large number
//
//float* a; //float* of [size] allocated on the GPU
// float* b; //float* of [size] allocated on the GPU
// float* c; //flo
//
//cudaMallocManaged((void**)&a, sizeof(float) * sz);
//cudaMallocManaged((void**)&b, sizeof(float) * sz);
//cudaMallocManaged((void**)&c, sizeof(float) * sz);
//
//
//float t = omp_get_wtime();
//printf("\n Calculating... (BlackCat_Tensors) reps outside\n");
//
//for (int i = 0; i < reps; ++i) {
//add_kernel<<<blocks(sz), threads()>>>(a, b, c, sz);
//cudaDeviceSynchronize();
//}
//t = omp_get_wtime() - t;
//printf("It took me %f clicks (%f seconds).\n", t, ((float) t));
//
//
// t = omp_get_wtime();
//printf("\n Calculating... (BlackCat_Tensors) reps inside \n");
//
//add_kernel<<<blocks(sz), threads()>>>(a, b, c, sz, reps);
//cudaDeviceSynchronize();
//
//
//t = omp_get_wtime() - t;
//printf("It took me %f clicks (%f seconds).\n", t, ((float) t));
//
//
//
//
//
// cudaFree(a);
// cudaFree(b);
// cudaFree(c);
//}
//
//
//#endif /* TEST_AREA_CU_ */
