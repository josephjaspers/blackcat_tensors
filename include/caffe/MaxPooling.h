/*
 * MaxPooling.h
 *
 *	Created on: Nov 10, 2019
 *			Author: joseph
 */

#ifndef BLACKCAT_TENSORS_CAFFE_MAXPOOLING_H_
#define BLACKCAT_TENSORS_CAFFE_MAXPOOLING_H_

/*
 * THIS IS NOT AN ORIGINAL CAFFE FILE.
 * MAXPOOLING IMPLEMENTATION WAS ORIGINALLY CREATED BY THE CAFFE AUTHOR(S)
 *
 * Modifications to the indexing and how channels are handled.
 * The data is also in col-major format despite the variables being the same.
 *
 */

namespace BC {

class host_tag;

namespace caffe {

template <class Dtype>
void MaxPoolForward(
		BC::host_tag,
		const Dtype* img_data,
		const int num,
		const int channels,
		const int height,   const int width,
		const int pool_h,   const int pool_w,
		const int krnl_h,   const int krnl_w,
		const int stride_h, const int stride_w,
		const int pad_h,    const int pad_w,
		Dtype* out_data, int* mask)
{
	using std::min;
	using std::max;

	int img_size = height * width;
	int pool_size = pool_h * pool_w;

	int img_total_size = img_size * channels;
	int pool_total_size = pool_size * channels;

	for (int n = 0; n < num; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int ph = 0; ph < pool_h; ++ph) {
				for (int pw = 0; pw < pool_w; ++pw) {
					int hstart = ph * stride_h - pad_h;
					int wstart = pw * stride_w - pad_w;
					int hend = min(hstart + krnl_h, height);
					int wend = min(wstart + krnl_w, width);
					hstart = max(hstart, 0);
					wstart = max(wstart, 0);
					const int pool_index = ph * pool_w + pw + c * pool_size;
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							const int index = h * width + w + c * img_size;
							if (img_data[index] > out_data[pool_index]) {
								out_data[pool_index] = img_data[index];
								mask[pool_index] = index;
							}
						}
					}
				}
			}
		}

		img_data += img_total_size;
		out_data += pool_total_size;
		mask += pool_total_size;
	}
}

template <typename Dtype>
void MaxPoolBackward(
		BC::host_tag,
		const Dtype* top_diff, const int* mask,
		const int num,      const int channels,
		const int height,   const int width,
		const int pool_h,   const int pool_w,
		const int krnl_h,   const int krnl_w,
		const int stride_h, const int stride_w,
		const int pad_h,    const int pad_w,
		Dtype* bottom_diff) {

	int pool_size = pool_h * pool_w;
	int data_size = height * width;

	for (int n = 0; n < num; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int ph = 0; ph < pool_h; ++ph) {
				for (int pw = 0; pw < pool_w; ++pw) {
					const int pool_index = ph * pool_w + pw + c * pool_size;
					const int bottom_index = mask[pool_index];
					bottom_diff[bottom_index] += top_diff[pool_index];
				}
			}
		}

		bottom_diff += data_size * channels;
		top_diff += pool_size * channels;
		mask += pool_size * channels;
	}
}

}
}

#endif /* MAXPOOLING_H_ */
