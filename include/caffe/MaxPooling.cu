/*
 * MaxPooling.h
 *
 *	Created on: Nov 10, 2019
 *			Author: joseph
 */

#ifdef __CUDACC__
#ifndef BLACKCAT_TENSORS_CAFFE_MAXPOOLING_CU_
#define BLACKCAT_TENSORS_CAFFE_MAXPOOLING_CU_


#include "Caffe_Cuda.h"

/*
 * THIS IS NOT AN ORIGINAL CAFFE FILE.
 * MAXPOOLING IMPLEMENTATION WAS ORIGINALLY CREATED BY THE CAFFE AUTHOR(S)
 */

namespace BC {
namespace caffe {

using BC::traits::min;
using BC::traits::max;

template <typename Dtype> __global__
void MaxPoolForward_gpu_kernel(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width, const int pooled_height,
		const int pooled_width, const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data, int* mask)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int pw = index % pooled_width;
		const int ph = (index / pooled_width) % pooled_height;
		const int c = (index / pooled_width / pooled_height) % channels;
		const int n = index / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		const int hend = min(hstart + kernel_h, height);
		const int wend = min(wstart + kernel_w, width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		Dtype maxval = -FLT_MAX;
		int maxidx = -1;
		const Dtype* const bottom_slice =
				bottom_data + (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				if (bottom_slice[h * width + w] > maxval) {
					maxidx = h * width + w;
					maxval = bottom_slice[maxidx];
				}
			}
		}
		top_data[index] = maxval;
		mask[index] = maxidx;
	}
}


template <typename Dtype>
__global__ void MaxPoolBackward_gpu_kernel(
		const int nthreads,
		const Dtype* const top_diff,
		const int* const mask, const int num,
		const int channels, const int height, const int width,
		const int pooled_height, const int pooled_width, const int kernel_h,
		const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
		const int pad_w, Dtype* const bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// find out the local index
		// find out the local offset
		const int w = index % width;
		const int h = (index / width) % height;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;
		const int phstart =
				 (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
		const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
		const int pwstart =
				 (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
		const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
		Dtype gradient = 0;
		const int offset = (n * channels + c) * pooled_height * pooled_width;
		const Dtype* const top_diff_slice = top_diff + offset;
			const int* const mask_slice = mask + offset;
			for (int ph = phstart; ph < phend; ++ph) {
				for (int pw = pwstart; pw < pwend; ++pw) {
					if (mask_slice[ph * pooled_width + pw] == h * width + w) {
						gradient += top_diff_slice[ph * pooled_width + pw];
					}
				}
			}
		bottom_diff[index] = gradient;
	}
}



template <typename Dtype>
void MaxPoolForward(
		BC::device_tag,
		const Dtype* const bottom_data,
		const int num, const int channels,
		const int height, const int width,
		const int pooled_height, const int pooled_width,
		const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w,
		const int pad_h, const int pad_w,
		Dtype* const top_data, int* mask)
{

	int size = pooled_height * pooled_width * channels * num;
	MaxPoolForward_gpu_kernel<<<CAFFE_GET_BLOCKS(size), CAFFE_CUDA_NUM_THREADS>>>(
			size, bottom_data,
			num, channels,
			height, width,
			pooled_height, pooled_width,
			kernel_h, kernel_w,
			stride_h, stride_w,
			pad_h, pad_w,
			top_data, mask);
}

template <typename Dtype>
void MaxPoolBackward(
		BC::device_tag,
		const Dtype* const top_diff, const int* const mask,
		const int num,      const int channels,
		const int height,   const int width,
		const int pooled_height, const int pooled_width,
		const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w,
		const int pad_h,    const int pad_w,
		Dtype* bottom_diff)
{
	int size = pooled_height * pooled_width * channels * num;
	MaxPoolBackward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(size), CAFFE_CUDA_NUM_THREADS>>>(
			size, top_diff, mask,
			num, channels,
			height, width,
			pooled_height, pooled_width,
			kernel_h, kernel_w,
			stride_h, stride_w,
			pad_h, pad_w,
			bottom_diff);
}

}
}

#endif /* MAXPOOLING_H_ */
#endif // __CUDACC__
