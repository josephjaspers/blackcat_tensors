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

template<class OutputTensorDescriptor>
class Layer_Input_Base;

template<class OutputTensorDescriptor>
struct Layer_Output_Base
{
	using output_value_type = typename OutputTensorDescriptor::value_type;
	using output_system_tag = typename OutputTensorDescriptor::system_tag;
	using output_allocator_type = typename OutputTensorDescriptor::allocator_type;
	using output_tensor_dim = typename OutputTensorDescriptor::tensor_dim;
	using output_shape_type = bc::Dim<output_tensor_dim::value>;
	using output_tensor_type = typename OutputTensorDescriptor::type;
	using batched_output_tensor_type = typename OutputTensorDescriptor::batched_type;
	using next_layer_type = Layer_Input_Base<OutputTensorDescriptor>;

private:
	using self_type = Layer_Output_Base<OutputTensorDescriptor>;

protected:
	next_layer_type* m_next_layer = nullptr;
	bc::Dim<output_tensor_dim::value> m_output_shape;

public:
//	virtual const output_tensor_type& get_output() const=0;
	void set_next(next_layer_type& next) { m_next_layer = &next; }
	output_shape_type output_shape() const { return m_output_shape; }
	const next_layer_type& next_layer() const { return *m_next_layer; }
	next_layer_type& next_layer() { return *m_next_layer; }
	virtual ~Layer_Output_Base() {}
};

template<class InputTensorDescriptor>
class Layer_Input_Base
{
	using input_value_type = typename InputTensorDescriptor::value_type;
	using input_system_tag = typename InputTensorDescriptor::system_tag;
	using input_allocator_type = typename InputTensorDescriptor::allocator_type;
	using input_tensor_dim = typename InputTensorDescriptor::tensor_dim;
	using input_shape_type = bc::Dim<input_tensor_dim::value>;
	using input_tensor_type = typename InputTensorDescriptor::type;
	using batched_input_tensor_type = typename InputTensorDescriptor::batched_type;
	using prev_layer_type = Layer_Output_Base<InputTensorDescriptor>;
private:
	using self_type = Layer_Input_Base<InputTensorDescriptor>;

protected:
	prev_layer_type* m_prev_layer;
	input_shape_type m_input_shape;

public:
//	virtual const input_tensor_type& get_input() const=0;
	void set_prev(prev_layer_type& prev) { m_prev_layer = &prev; }
	input_shape_type input_shape() const { return m_input_shape; }
	const prev_layer_type& prev_layer() const { return *m_prev_layer; }
	prev_layer_type& prev_layer() { return *m_prev_layer; }
	virtual ~Layer_Input_Base() {}
};


template<
	class DerivedLayer,
	class InputTensorDescriptor,
	class OutputTensorDescriptor=InputTensorDescriptor>
struct Layer_Base:
		Layer_Input_Base<InputTensorDescriptor>,
		Layer_Output_Base<OutputTensorDescriptor>
{
	using value_type = typename InputTensorDescriptor::value_type;
	using system_tag = typename InputTensorDescriptor::system_tag;
	using allocator_type = typename InputTensorDescriptor::allocator_type;
	using input_tensor_dim = typename InputTensorDescriptor::tensor_dim;

	using shape_type = bc::Dim<input_tensor_dim::value>;
	using input_tensor_type = typename InputTensorDescriptor::type;
	using batched_input_tensor_type = typename InputTensorDescriptor::batched_type;

	using output_value_type = typename OutputTensorDescriptor::value_type;
	using output_system_tag = typename OutputTensorDescriptor::system_tag;
	using output_allocator_type = typename OutputTensorDescriptor::allocator_type;
	using output_tensor_dim = typename OutputTensorDescriptor::tensor_dim;
	using output_shape_type = bc::Dim<output_tensor_dim::value>;
	using output_tensor_type = typename OutputTensorDescriptor::type;
	using batched_output_tensor_type = typename OutputTensorDescriptor::batched_type;

	static constexpr value_type default_learning_rate = .01;

private:
	using traits = layer_traits<DerivedLayer>;

	value_type m_learning_rate = default_learning_rate;
	bc::size_t m_batch_sz=1;
	std::string m_classname;

protected:
	shape_type m_input_shape;
	output_shape_type m_output_shape;

public:
	/**
	 * m_classname should be initialized by supplying `__func__` to the first argument
	 * of the Layer_Base. `parse_classname()` will normalize the string
	 * as `__func__` is compiler dependent.
	 */
	Layer_Base(
			std::string classname,
			shape_type input_shape,
			output_shape_type output_shape):
		m_classname(parse_classname(classname)),
		m_input_shape(input_shape),
		m_output_shape(output_shape) {
	}

	Layer_Base(std::string classname, shape_type input_shape):
			m_classname(parse_classname(classname)),
			m_input_shape(input_shape) {}

	virtual ~Layer_Base() {}

	virtual output_shape_type get_output_shape() const { return m_output_shape; }
	virtual shape_type get_input_shape() const { return m_input_shape; }

	auto get_batched_input_shape() const { return m_input_shape.concat(m_batch_sz); }
	auto get_batched_output_shape() const { return m_output_shape.concat(m_batch_sz); }

	bc::size_t input_size() const { return this->m_input_shape.prod(); }
	bc::size_t output_size() const { return this->m_output_shape.prod(); }
	bc::size_t batch_size() const { return m_batch_sz; }

	bc::size_t batched_input_size() const { return input_size() * m_batch_sz; }
	bc::size_t batched_output_size() const { return output_size() * m_batch_sz; }

	void set_batch_size(int batch_size)
	{
		m_batch_sz = batch_size;
		set_batch_size_hook(batch_size);
	}

	virtual void set_batch_size_hook(int batch_size) {}

	void set_learning_rate(value_type learning_rate)
	{
		m_learning_rate = learning_rate;
		set_learning_rate_hook(learning_rate);
	}

	virtual void set_learning_rate_hook(value_type learning_rate) {}

	auto get_learning_rate() const { return m_learning_rate; }
	auto get_batched_learning_rate() const { return m_learning_rate / m_batch_sz; }

	virtual void save(Layer_Loader&) const {};
	virtual void save_from_cache(Layer_Loader&, const Cache&) const {}
	virtual void load(Layer_Loader&) {};
	virtual void load_to_cache(Layer_Loader&, const Cache&) {}

	void copy_training_data_to_single_predict(Cache&, int batch_index) {}
	void update_weights() {}
	void clear_bp_storage(Cache&) {}

	/**Returns the derived_classes class namepse.
	 * Note: Architecture dependent
	 */
	const std::string& classname() const { return m_classname; }
private:
	template<class T>
	using query_optimizer_type = typename T::optimizer_type;

public:
	template<int ADL=0>
	std::string get_string_architecture() const {
		std::string yaml = classname() + ':'
			+ "\n\tinput_shape: " + get_input_shape().to_string();

		using optimizer_type = bc::traits::conditional_detected_t<
			query_optimizer_type, DerivedLayer, bc::traits::None>;

		if (!std::is_same<optimizer_type, bc::traits::None>::value) {
			auto opt_name = bc_get_classname_of(optimizer_type());
			yaml += "\n\toptimizer: ";
			yaml += parse_classname(opt_name);
		}

		std::string other_features = get_string_architecture_hook();
		if (other_features != "")
			yaml += "\n\t" + other_features;

		return yaml;
	}

	virtual std::string get_string_architecture_hook() const { return ""; }


	static std::string parse_classname(std::string classname) {
		auto classname_ns = std::find(classname.rbegin(), classname.rend(), ':');
		classname.erase(classname.rend().base(), classname_ns.base());
		return classname;
	}

	auto default_input_tensor_factory() const {
		return [&]() {
			return input_tensor_type(get_input_shape()).zero();
		};
	}

	auto default_output_tensor_factory() const {
		return [&]() {
			return output_tensor_type(get_output_shape()).zero();
		};
	}

	auto default_batched_input_tensor_factory() const {
		return [&]() {
			return batched_input_tensor_type(
					get_batched_input_shape()).zero();
		};
	}

	auto default_batched_output_tensor_factory() const {
		return [&]() {
			return batched_output_tensor_type(
					get_batched_output_shape()).zero();
		};
	}
};

}
}



#endif /* LAYER_H_ */
