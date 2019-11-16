/*
 * Convolution.h
 *
 *  Created on: Aug 27, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_CONVOLUTION_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_CONVOLUTION_H_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class IsRecurrent=std::false_type>
struct Convolution:
		public Layer_Base<
				Convolution<SystemTag, ValueType, IsRecurrent>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using parent_type = Layer_Base<
			Convolution<SystemTag, ValueType, IsRecurrent>>;

	using mat = BC::Matrix<value_type, allocator_type>;
	using tensor4 = BC::Tensor<4, value_type, allocator_type>;
	using cube = BC::Cube<value_type, allocator_type>;

	using greedy_evaluate_delta = std::true_type;
	using input_tensor_dimension = BC::traits::Integer<3>;
	using output_tensor_dimension = BC::traits::Integer<3>;
	using requires_extra_cache = std::true_type;

	using is_recurrent = IsRecurrent;

private:

	allocator_type m_allocator;

	mat w;  //dims=[kernel_size, numb_kernels]
	mat w_gradients;

	Dim<3> m_krnl_shape; //dims=[rows, cols, numb_kernels]
	Dim<2> m_padding;
	Dim<2> m_strides;
	Dim<2> m_dilation;

	Dim<3> m_input_shape;
	Dim<3> m_output_shape;

	Dim<2> m_column_image_shape;

	using col_image_key = BC::nn::cache_key<
			BC::utility::Name<'c','x'>, cube, is_recurrent>;

public:

	Dim<3> get_input_shape() const { return m_input_shape; }
	Dim<3> get_output_shape() const { return m_output_shape; }

	Dim<3> get_batched_column_image_shape() const {
		return m_column_image_shape.concat(this->batch_size());
	}

	Dim<4> get_kernel_shape() const {
		return BC::Dim<4> {
				m_krnl_shape[0], m_krnl_shape[1],
				m_input_shape[2], m_krnl_shape[2] };
	}

	Convolution(
			Dim<3> img_dims,
			Dim<3> krnl_dims,
			Dim<2> padding=Dim<2>().fill(0),
			Dim<2> strides=Dim<2>().fill(1),
			Dim<2> dilation=Dim<2>().fill(1)):
		parent_type(__func__),
		w(krnl_dims.prod(2)*img_dims[2], krnl_dims[2]),
		w_gradients(w.get_shape()),
		m_input_shape(img_dims),
		m_krnl_shape(krnl_dims),
		m_padding(padding),
		m_strides(strides),
		m_dilation(dilation)
	{
		auto out_dim = [&](int dim_idx) {
			return (m_input_shape[dim_idx] +
					m_padding[dim_idx]*2 -
					m_krnl_shape[dim_idx]) /
					m_strides[dim_idx] + 1;
		};

		m_output_shape = BC::dim(out_dim(0), out_dim(1), krnl_dims[2]);
		m_column_image_shape = BC::dim(m_krnl_shape.prod(2) * img_dims[2], out_dim(0) * out_dim(1));

		this->m_input_sz = m_input_shape.size();
		this->m_output_sz = m_output_shape.size();

		w.randomize(-1, 1);
	}


	template<class X>
	auto forward_propagation(const X& x, Cache& cache) {
		tensor4 y(this->get_batched_output_shape());
		cube col_x(get_batched_column_image_shape());

		for (int i = 0; i < this->batch_size(); ++i) {
			BC::im2col(x.get_stream(),
					col_x[i].internal(),
					x[i].internal(),
					m_krnl_shape,
					m_padding,
					m_strides,
					m_dilation);

			BC::Dim<2> mat_y_shape = {y.rows() * y.cols(), w.cols() };
			y[i].reshaped(mat_y_shape) = col_x[i].t() * w;
		}

		cache.store(col_image_key(), col_x);
		return y;
	}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy, Cache& cache) {
		static_assert(Delta::tensor_dimension==4, "3d delta");
		static_assert(X::tensor_dimension==4, "4d X");

		cube& col_x = cache.load(col_image_key());

		BC_omp_for__
		for (int i = 0; i < this->batch_size(); ++i) {
			auto mat_dy = reshape(dy[i], shape(dy.rows() * dy.cols(), w.cols()));
			w_gradients -= col_x[i] * mat_dy;
		}

		return w.reshaped(get_kernel_shape()).multichannel_conv2d_data_backwards(dy);
	}

	void update_weights() {
		w += w_gradients * this->get_batched_learning_rate();
		w_gradients.zero();
	}

	void save(Layer_Loader& loader) {
		loader.save_variable(w, "w");
	}

	void load(Layer_Loader& loader) {
		loader.load_variable(w, "w");
	}
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag, bool isRecurrent=false>
auto convolution(
		SystemTag system_tag,
		Dim<3> img_dims,
		Dim<3> krnl_dims,
		Dim<2> padding=Dim<2>().fill(0),
		Dim<2> strides=Dim<2>().fill(1),
		Dim<2> dilation=Dim<2>().fill(1))
{
	return Convolution<
			SystemTag,
			ValueType,
			BC::traits::truth_type<isRecurrent>>(
					img_dims,
					krnl_dims,
					padding,
					strides,
					dilation);
}
#endif

template<
		class SystemTag=BLACKCAT_DEFAULT_SYSTEM_T,
		bool isRecurrent=false>
auto convolution(
		SystemTag system_tag,
		Dim<3> img_dims,
		Dim<3> krnl_dims,
		Dim<2> padding=Dim<2>().fill(0),
		Dim<2> strides=Dim<2>().fill(1),
		Dim<2> dilation=Dim<2>().fill(1))
{
	using value_type = typename SystemTag::default_floating_point_type;
	return Convolution<
			SystemTag,
			value_type,
			BC::traits::truth_type<isRecurrent>>(
					img_dims,
					krnl_dims,
					padding,
					strides,
					dilation);
}

}
}



#endif /* CONVOLUTION_H_ */
