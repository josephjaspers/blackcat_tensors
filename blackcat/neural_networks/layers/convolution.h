/*
 * Convolution.h
 *
 *  Created on: Aug 27, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_CONVOLUTION_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_CONVOLUTION_H_

#include "layer_base.h"

namespace bc {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class Optimizer=Stochastic_Gradient_Descent>
struct Convolution:
		public Layer_Base<
				Convolution<SystemTag, ValueType, Optimizer>,
				Tensor_Descriptor<ValueType, SystemTag, bc::traits::Integer<3>>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using optimizer_type = Optimizer;

	using self_type = Convolution<SystemTag, ValueType, Optimizer>;
	using input_descriptor_t = Tensor_Descriptor<ValueType, SystemTag, bc::traits::Integer<3>>;
	using parent_type = Layer_Base<self_type, input_descriptor_t>;

	using input_tensor_dim = bc::traits::Integer<3>;
	using output_tensor_dim = bc::traits::Integer<3>;

	using greedy_evaluate_delta = std::true_type;
	using requires_extra_cache = std::true_type;

	using defines_single_predict = std::true_type;

private:

	using mat = bc::Matrix<value_type, allocator_type>;
	using tensor4 = bc::Tensor<4, value_type, allocator_type>;
	using cube = bc::Cube<value_type, allocator_type>;

	using mat_opt_t = typename Optimizer::template Optimizer<mat>;

	allocator_type m_allocator;

	mat w;  //dims=[kernel_size, numb_kernels]
	mat w_gradients;
	mat_opt_t w_opt;

	Dim<3> m_krnl_shape; //dims=[rows, cols, numb_kernels]
	Dim<2> m_padding;
	Dim<2> m_strides;
	Dim<2> m_dilation;
	Dim<2> m_column_image_shape;

	using col_image_key = bc::nn::cache_key<bc::utility::Name<'c','x'>, cube>;

public:

	Convolution(
			Dim<3> img_dims,
			Dim<3> krnl_dims,
			Dim<2> padding=Dim<2>().fill(0),
			Dim<2> strides=Dim<2>().fill(1),
			Dim<2> dilation=Dim<2>().fill(1)):
		parent_type(__func__, img_dims),
		w(krnl_dims.prod(2)*img_dims[2], krnl_dims[2]),
		w_gradients(w.get_shape()),
		w_opt(w.get_shape()),
		m_krnl_shape(krnl_dims),
		m_padding(padding),
		m_strides(strides),
		m_dilation(dilation)
	{
		auto out_dim = [&](int dim_idx) {
			return (this->m_input_shape[dim_idx] +
					m_padding[dim_idx]*2 -
					m_krnl_shape[dim_idx]) /
					m_strides[dim_idx] + 1;
		};

		this->m_output_shape = bc::dim(out_dim(0), out_dim(1), krnl_dims[2]);
		m_column_image_shape = bc::dim(
						m_krnl_shape.prod(2) * img_dims[2],
						out_dim(0) * out_dim(1));

		w.randomize(-1, 1);
	}


	template<class X>
	auto forward_propagation(const X& x, Cache& cache)
	{
		tensor4 y(this->get_batched_output_shape());
		cube col_x(get_batched_column_image_shape());
		col_x.zero();

		for (int i = 0; i < this->batch_size(); ++i) {
			bc::im2col(x.get_stream(),
					col_x[i].expression_template(),
					x[i].expression_template(),
					m_krnl_shape,
					m_padding,
					m_strides,
					m_dilation);

			bc::Dim<2> mat_y_shape = {y.rows() * y.cols(), w.cols() };
			y[i].reshaped(mat_y_shape) = col_x[i].t() * w;
		}

		cache.store(col_image_key(), col_x);
		return y;
	}

	template<class X>
	auto single_predict(const X& x, Cache& cache)
	{
		cube y(this->get_output_shape());
		mat col_x(m_column_image_shape);
		col_x.zero();

		bc::im2col(x.get_stream(),
				col_x.expression_template(),
				x.expression_template(),
				m_krnl_shape,
				m_padding,
				m_strides,
				m_dilation);

			bc::Dim<2> mat_y_shape = {y.rows() * y.cols(), w.cols() };
			y.reshaped(mat_y_shape) = col_x.t() * w;

		return y;
	}


	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy, Cache& cache)
	{
		cube& col_x = cache.load(col_image_key());
		tensor4 delta_dx(this->get_batched_input_shape());
		delta_dx.zero();
		mat mat_delta_dx(m_column_image_shape);

		for (int i = 0; i < this->batch_size(); ++i) {
			auto mat_dy = dy[i].reshaped(w.cols(), dy.rows() * dy.cols());
			w_gradients -= col_x[i] * mat_dy.t();
			mat_delta_dx = w * mat_dy;
			bc::col2im(x.get_stream(),
					mat_delta_dx.expression_template(),
					delta_dx[i].expression_template(),
					m_krnl_shape,
					m_padding,
					m_strides,
					m_dilation);
		}
		return delta_dx;
	}

	Dim<3> get_batched_column_image_shape() const {
		return m_column_image_shape.concat(this->batch_size());
	}

	Dim<4> get_kernel_shape() const {
		return bc::Dim<4> {
			m_krnl_shape[0], m_krnl_shape[1],
			this->m_input_shape[2], m_krnl_shape[2]
		};
	}

	void update_weights()
	{
		w_opt.update(w, w_gradients);
		w_gradients.zero();
	}

	void set_learning_rate(value_type lr)
	{
		parent_type::set_learning_rate(lr);
		w_opt.set_learning_rate(this->get_batched_learning_rate());
	}

	virtual void save(Layer_Loader& loader) const override
	{
		loader.save_variable(w, "w");
		w_opt.save(loader, "w_opt");
	}

	virtual void load(Layer_Loader& loader) override
	{
		loader.load_variable(w, "w");
		w_opt.load(loader, "w_opt");
	}

	auto& get_weights() const { return w; }
	auto& get_weights()       { return w; }


};


template<
		class SystemTag=BLACKCAT_DEFAULT_SYSTEM_T,
		class Optimizer=nn_default_optimizer_type>
auto convolution(
		SystemTag system_tag,
		Dim<3> img_dims,
		Dim<3> krnl_dims,
		Dim<2> padding=Dim<2>().fill(0),
		Dim<2> strides=Dim<2>().fill(1),
		Dim<2> dilation=Dim<2>().fill(1),
		Optimizer=Optimizer())
{
	using value_type = typename SystemTag::default_floating_point_type;
	return Convolution<
			SystemTag,
			value_type,
			Optimizer>(
					img_dims,
					krnl_dims,
					padding,
					strides,
					dilation);
}


template<
		class SystemTag=BLACKCAT_DEFAULT_SYSTEM_T,
		class Optimizer=nn_default_optimizer_type>
auto recurrent_convolution(
		SystemTag system_tag,
		Dim<3> img_dims,
		Dim<3> krnl_dims,
		Dim<2> padding=Dim<2>().fill(0),
		Dim<2> strides=Dim<2>().fill(1),
		Dim<2> dilation=Dim<2>().fill(1),
		Optimizer=Optimizer())
{
	using value_type = typename SystemTag::default_floating_point_type;
	return Convolution<
			SystemTag,
			value_type,
			Optimizer>(
					img_dims,
					krnl_dims,
					padding,
					strides,
					dilation);
}


template<
		class SystemTag,
		class Optimizer>
auto convolution(
		SystemTag system_tag,
		Dim<3> img_dims,
		Dim<3> krnl_dims,
		Optimizer,
		Dim<2> padding=Dim<2>().fill(0),
		Dim<2> strides=Dim<2>().fill(1),
		Dim<2> dilation=Dim<2>().fill(1))
{
	using value_type = typename SystemTag::default_floating_point_type;
	return Convolution<
			SystemTag,
			value_type,
			Optimizer>(
					img_dims,
					krnl_dims,
					padding,
					strides,
					dilation);
}


template<
		class SystemTag,
		class Optimizer>
auto recurrent_convolution(
		SystemTag system_tag,
		Dim<3> img_dims,
		Dim<3> krnl_dims,
		Optimizer,
		Dim<2> padding=Dim<2>().fill(0),
		Dim<2> strides=Dim<2>().fill(1),
		Dim<2> dilation=Dim<2>().fill(1))
{
	using value_type = typename SystemTag::default_floating_point_type;
	return Convolution<
			SystemTag,
			value_type,
			Optimizer>(
					img_dims,
					krnl_dims,
					padding,
					strides,
					dilation);
}


}
}



#endif /* CONVOLUTION_H_ */
