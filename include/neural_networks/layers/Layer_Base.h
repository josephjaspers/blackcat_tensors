 /*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *	  Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "../Layer_Loader.h"
#include "Layer_Traits.h"
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <ostream>

namespace BC {
namespace nn {

template<class DerivedLayer>
class Layer_Base {

	using traits = layer_traits<DerivedLayer>;

	std::string m_classname;
	std::string m_directory_save_path;
	std::string m_additional_architecture_features;

	auto& as_derived() const {
		return static_cast<const DerivedLayer&>(*this);
	}
	auto& as_derived() {
		return static_cast<DerivedLayer&>(*this);
	}

public:

	static constexpr double default_learning_rate = .01;

protected:

	BC::size_t m_input_sz;
	BC::size_t m_output_sz;
	BC::size_t m_batch_sz;
	double m_learning_rate = default_learning_rate;

public:
	/**
	 * m_classname should be initialized by supplying `__func__` to the first argument
	 * of the Layer_Base. `parse_classname()` will normalize the string
	 * as `__func__` is compiler dependent.
	 */
	Layer_Base(std::string classname_, BC::size_t inputs=0, BC::size_t outputs=0):
		m_classname(parse_classname(classname_)),
		m_input_sz(inputs),
		m_output_sz(outputs),
		m_batch_sz(1) {}

	void resize(BC::size_t inputs, BC::size_t outputs) {
		m_input_sz = inputs;
		m_output_sz = outputs;
	}

	/**Returns the derived_classes class name.
	 * Note: Architecture dependent
	 */
	std::string classname() const { return m_classname; }
	std::string get_string_architecture() const {
		std::string yaml =  classname() + ':'
				+ "\n\tinputs: " + std::to_string(input_size())
				+ "\n\toutputs: " + std::to_string(output_size());

		if (m_additional_architecture_features != "") {
			yaml += "\n\t" + m_additional_architecture_features;
		}

		return yaml;
	}

	/**
	 * Add additional features to be stored in the yaml 'architecture' string.
	 * This is a 'reserved' method. It will be used for user layers or more advanced layers
	 * in the future.
	 */
	void add_architecture_features(std::string features) {
		m_additional_architecture_features+= features;
	}

	void clear_architecture_features() {
		m_additional_architecture_features = "";
	}

	///get_shape must be shadowed (over-ridden) for Convolution/layers that expect non-vector input/outputs
	auto get_input_shape() const { return BC::Dim<1>{m_input_sz}; }
	auto get_output_shape() const { return BC::Dim<1>{m_output_sz}; }

	auto get_batched_input_shape() const {
		return as_derived().get_input_shape().concat(m_batch_sz);
	}

	auto get_batched_output_shape() const {
		return as_derived().get_output_shape().concat(m_batch_sz);
	}

	BC::size_t input_size() const { return m_input_sz; }
	BC::size_t output_size() const { return m_output_sz; }
	BC::size_t batch_size() const { return m_batch_sz; }

	BC::size_t batched_input_size() const { return m_input_sz * m_batch_sz; }
	BC::size_t batched_output_size() const { return m_output_sz * m_batch_sz; }

	void set_batch_size(int bs) { m_batch_sz = bs;}

	void set_learning_rate(double lr) { m_learning_rate = lr; }
	auto get_learning_rate() const { return m_learning_rate; }
	auto get_batched_learning_rate() const { return m_learning_rate / m_batch_sz; }
	void update_weights() {}
	void clear_bp_storage(Cache&) {}

	void save(Layer_Loader&) {};
	void save_from_cache(Layer_Loader&, Cache&) {}
	void load(Layer_Loader&) {};
	void load_to_cache(Layer_Loader&, Cache&) {}

	void copy_training_data_to_single_predict(Cache&, int batch_index) {}

	static std::string parse_classname(std::string classname) {
		auto classname_ns = std::find(classname.rbegin(), classname.rend(), ':');
		classname.erase(classname.rend().base(), classname_ns.base());
		return classname;
	}


	template<int ADL=0>
	auto default_input_tensor_factory() const {
		using dimension      = typename traits::input_tensor_dimension;
		using value_type     = typename traits::value_type;
		using allocator_type = typename traits::allocator_type;

		return [&]() {
			return BC::Tensor<dimension::value, value_type, allocator_type>(
					get_input_shape()).zero();
		};
	}

	template<int ADL=0>
	auto default_output_tensor_factory() const {
		using dimension      = typename traits::output_tensor_dimension;
		using value_type     = typename traits::value_type;
		using allocator_type = typename traits::allocator_type;

		return [&]() {
			return BC::Tensor<dimension::value, value_type, allocator_type>(
					get_output_shape()).zero();
		};
	}

	template<int ADL=0>
	auto default_batched_input_tensor_factory() const {
		using dimension      = typename traits::input_tensor_dimension;
		using value_type     = typename traits::value_type;
		using allocator_type = typename traits::allocator_type;

		return [&]() {
			return BC::Tensor<dimension::value+1, value_type, allocator_type>(
				get_batched_input_shape()).zero();
		};
	}

	template<int ADL=0>
	auto default_batched_output_tensor_factory() const {
		using dimension      = typename traits::output_tensor_dimension;
		using value_type     = typename traits::value_type;
		using allocator_type = typename traits::allocator_type;

		return [&]() {
			return BC::Tensor<dimension::value+1, value_type, allocator_type>(
					get_batched_output_shape()).zero();
		};
	}

};

}
}



#endif /* LAYER_H_ */
