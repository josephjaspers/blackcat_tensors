 /*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *	  Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "../layer_loader.h"
#include "layer_traits.h"
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <ostream>

namespace bc {
namespace nn {

template<class LearningRateValueType>
struct network_runtime_traits
{
	using value_type = LearningRateValueType;
	value_type m_learning_rate;
	int m_batch_size;
};

template<class SystemTag, class ValueType>
using layer_default_allocator =
		bc::allocators::Polymorphic_Allocator<SystemTag, ValueType>;

template<
	class Dimension,
	class ValueType,
	class SystemTag,
	class Allocator=layer_default_allocator<SystemTag, ValueType>,
	class OutputDimension=Dimension,
	class OutputValueType=ValueType,
	class OutputSystemTag=SystemTag,
	class OutputAllocator=Allocator>
struct Layer_Base
{
	using allocator_type = Allocator;
	using value_type = ValueType;
	using system_tag = SystemTag;

	using input_tensor_dimension = Dimension;
	using batched_input_tensor_dimension = bc::traits::Integer<Dimension::value+1>;

	using output_tensor_dimension = OutputDimension;
	using batched_output_tensor_dimension = bc::traits::Integer<OutputDimension::value+1>;

	using output_value_type = OutputValueType;
	using output_system_tag = OutputSystemTag;
	using output_allocator_type = OutputAllocator;

	using next_layer_type = Layer_Base<
			output_tensor_dimension,
			output_value_type,
			output_system_tag>;

	using this_layer_type = Layer_Base<
			input_tensor_dimension,
			value_type,
			system_tag>;

	using input_tensor_type  = bc::Tensor< input_tensor_dimension::value, value_type, allocator_type>;
	using output_tensor_type = bc::Tensor<output_tensor_dimension::value, value_type, output_allocator_type>;

	using batched_input_tensor_type  = bc::Tensor<input_tensor_dimension::value  + 1, value_type, allocator_type>;
	using batched_output_tensor_type = bc::Tensor<output_tensor_dimension::value + 1, value_type, output_allocator_type>;

	using this_layer_pointer_type = std::shared_ptr<this_layer_type>;
	using next_layer_pointer_type = std::shared_ptr<next_layer_type>;

	using input_shape_type = bc::Dim<input_tensor_dimension::value>;
	using output_shape_type = bc::Dim<output_tensor_dimension::value>;

private:

	const std::string m_classname;
	std::shared_ptr<network_runtime_traits<value_type>> m_network_vars
		= std::shared_ptr<network_runtime_traits<value_type>>(
			new network_runtime_traits<value_type>());

	//post constructor initialization
	virtual void init() = 0;

protected:

	std::shared_ptr<allocator_type> m_allocator;
	std::weak_ptr<this_layer_type> m_input_layer;
	std::shared_ptr<next_layer_type> m_output_layer;

	input_shape_type m_input_shape;
	output_shape_type m_output_shape; //ptr??

public:

	Layer_Base(std::string classname):
		m_classname(parse_classname(classname)) {}

	batched_output_tensor_type y;

	virtual ~Layer_Base()=default;

	template<int ADL=0>
	std::string get_string_architecture() const {
		return classname() + ':'
			+ "\n\tinput_shape: " + m_input_shape.to_string();
	}

	virtual batched_output_tensor_type forward_propagation(
			const batched_input_tensor_type&  inputs) = 0;

	virtual batched_input_tensor_type back_propagation(
			const batched_output_tensor_type& delta) = 0;

	batched_output_tensor_type fp(
			const batched_input_tensor_type&  inputs) {
		return forward_propagation(inputs);
	}

	batched_input_tensor_type bp(
			const batched_output_tensor_type& delta) {
		return back_propagation(delta);

	}

	void set_batch_size(int bs)
	{
		set_batch_size_hook(bs);
		m_network_vars->m_batch_size = bs;
		y = batched_output_tensor_type(this->batched_output_shape());
	}

	void set_learning_rate(double lr)
	{
		set_learning_rate_hook(lr);
		m_network_vars->m_learning_rate = lr;
	}

	std::shared_ptr<this_layer_type> prev() {
		return m_input_layer.lock();
	}

	std::shared_ptr<this_layer_type> prev() const {
		return m_input_layer.lock();
	}

	std::shared_ptr<next_layer_type> next() {
		return m_input_layer;
	}

	std::shared_ptr<next_layer_type> next() const {
		return m_output_layer;
	}

	void set_prev(this_layer_pointer_type prev_layer) {
		m_input_layer = prev_layer;
	}

	void set_next(next_layer_pointer_type next_layer) {
		m_output_layer = next_layer;
	}

	void link(next_layer_pointer_type& next_layer) {
		set_next(next_layer);
		next_layer.set_prev(next_layer);
	}



protected:

	virtual void set_batch_size_hook(int bs) {}
	virtual void set_learning_rate_hook(double lr) {}

public:

	auto batch_size() const { return m_network_vars->m_batch_size; }

	auto learning_rate() const         { return m_network_vars->m_learning_rate; }
	auto batched_learning_rate() const { return learning_rate() / batch_size();  }

	auto input_shape() const { return m_input_shape; }
	auto output_shape() const { return m_output_shape; }

	auto batched_input_shape()  const { return m_input_shape.concat(batch_size());  }
	auto batched_output_shape() const { return m_output_shape.concat(batch_size()); }

	std::string classname() const { return m_classname; }
	virtual void save(Layer_Loader&) const = 0;
	virtual void save_from_cache(Layer_Loader&, Cache&) const {}
	virtual void load(Layer_Loader&) = 0;
	virtual void load_to_cache(Layer_Loader&, Cache&) {}

	void copy_training_data_to_single_predict(Cache&, int batch_index) {}

	//push me
	static std::string parse_classname(std::string classname) {
		auto classname_ns = std::find(classname.rbegin(), classname.rend(), ':');
		classname.erase(classname.rend().base(), classname_ns.base());
		return classname;
	}
};

}
}



#endif /* LAYER_H_ */
