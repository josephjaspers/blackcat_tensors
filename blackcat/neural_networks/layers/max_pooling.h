/*
 * Max_Pooling.h
 *
 *  Created on: Nov 13, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURAL_NETWORKS_LAYERS_MAX_POOLING_H_
#define BLACKCAT_TENSORS_NEURAL_NETWORKS_LAYERS_MAX_POOLING_H_

#include "layer_base.h"

namespace bc {
namespace nn {

template<class SystemTag, class ValueType>
struct Max_Pooling:
		public Layer_Base<Max_Pooling<SystemTag, ValueType>>
{
	using input_tensor_dim = bc::traits::Integer<3>;
	using output_tensor_dim = bc::traits::Integer<3>;

	using system_tag = SystemTag;
	using value_type = ValueType;
	using self_type = Max_Pooling<SystemTag, ValueType>;
	using parent_type = Layer_Base<self_type>;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;

	using greedy_evaluate_delta = std::true_type;
	using requires_extra_cache = std::true_type;
	using defines_single_predict = std::true_type;

private:

	using index_allocator_type = nn_default_allocator_type<SystemTag, bc::size_t>;
	using index_tensor_type = bc::Cube<bc::size_t, index_allocator_type>;
	using batched_index_tensor_type = bc::Tensor<4, bc::size_t, index_allocator_type>;
	using tensor_type = bc::Cube<value_type, allocator_type>;
	using batched_tensor_type = bc::Tensor<4, value_type, allocator_type>;

	using index_key_type =  bc::nn::cache_key<
		bc::utility::Name<'i','d','x'>, batched_index_tensor_type>;

	Dim<3> m_img_dims;  //channel_width_height
	Dim<3> m_pool_dims;
	Dim<2> m_krnl_dims;
	Dim<2> m_padding;
	Dim<2> m_strides;

public:

	Max_Pooling(
			Dim<3> img_dims,
			Dim<2> krnl_dims={3,3},
			Dim<2> padding={0,0},
			Dim<2> strides={-1,-1}):
		parent_type(__func__),
		m_img_dims(img_dims),
		m_krnl_dims(krnl_dims),
		m_padding(padding),
		m_strides(strides == Dim<2>{-1,-1} ? krnl_dims : strides)
	{
		Dim<2> img_hw = img_dims.template subdim<0,2>();
		m_pool_dims = ((img_hw + padding*2)/m_strides ).concat(img_dims[2]);

		BC_ASSERT((m_img_dims > 0).all(), "Max_Pooling img_dims must be greater than 0");
		BC_ASSERT((m_krnl_dims > 0).all(), "Max_Pooling krnl_dims must be greater than 0");
		BC_ASSERT((m_strides > 0).all(), "Max_Pooling strides must be greater than 0");
		BC_ASSERT((m_padding >= 0).all(), "Max_Pooling krnl_dims must be greater than 0 or equal to 0");

		this->m_input_sz = m_img_dims.size();
		this->m_output_sz = m_pool_dims.size();
	}

	template<class Image>
	auto forward_propagation(const Image& image, Cache& cache)
	{
		batched_index_tensor_type mask(this->get_batched_output_shape());
		mask.zero();

		batched_tensor_type pooled_image(this->get_batched_output_shape());
		pooled_image.zero();

		bc::max_pooling_forward(
				bc::streams::select_on_get_stream(image),
				image.expression_template(),
				pooled_image.expression_template(),
				mask.expression_template(),
				m_krnl_dims,
				m_padding,
				m_strides);

		cache.store(index_key_type(), mask);
		return pooled_image;
	}

	template<class Image>
	auto single_predict(const Image& image, Cache& cache)
	{
		index_tensor_type mask(this->get_output_shape());
		mask.zero();

		tensor_type pooled_image(this->get_output_shape());
		pooled_image.zero();

		bc::max_pooling_forward(
				bc::streams::select_on_get_stream(image),
				image.expression_template(),
				pooled_image.expression_template(),
				mask.expression_template(),
				m_krnl_dims,
				m_padding,
				m_strides);

		return pooled_image;
	}

	template<class Image, class Delta>
	auto back_propagation(
			const Image& image,
			const Delta& pooled_delta,
			Cache& cache)
	{
		batched_index_tensor_type& mask = cache.load(index_key_type());
		batched_tensor_type delta_x(this->get_batched_input_shape());

		bc::max_pooling_backward(
				bc::streams::select_on_get_stream(image),
				delta_x.expression_template(),
				pooled_delta.expression_template(),
				mask.expression_template(),
				m_krnl_dims,
				m_padding,
				m_strides);

		return delta_x;
	}

	Dim<3> get_input_shape() const {
		return m_img_dims;
	}

	Dim<3> get_output_shape() const {
		return m_pool_dims;
	}
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag>
Max_Pooling<SystemTag, ValueType> max_pooling(
		SystemTag system_tag,
		Dim<3> img_dims,
		Dim<2> krnl_dims={3,3},
		Dim<2> padding={0,0},
		Dim<2> strides={-1,-1})
{
	return Max_Pooling<SystemTag, ValueType>(
			img_dims, krnl_dims, padding, strides);
}

#endif

template<class SystemTag>
auto max_pooling(
		SystemTag system_tag,
		Dim<3> img_dims,
		Dim<2> krnl_dims={3,3},
		Dim<2> padding={0,0},
		Dim<2> strides={-1,-1})
{
	return Max_Pooling<
			SystemTag,
			typename SystemTag::default_floating_point_type>(
					img_dims, krnl_dims, padding, strides);
}

auto max_pooling(
		Dim<3> img_dims,
		Dim<2> krnl_dims={3,3},
		Dim<2> padding={0,0},
		Dim<2> strides={-1,-1})
{
	return Max_Pooling<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(
					img_dims, krnl_dims, padding, strides);
}


}
}



#endif /* MAX_POOLING_H_ */
