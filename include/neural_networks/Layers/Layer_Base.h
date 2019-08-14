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

class Layer_Base {

	const std::string m_classname;
	std::string m_directory_save_path;
	std::string m_additional_architecture_features;

	BC::size_t m_input_sz;
	BC::size_t m_output_sz;
	BC::size_t m_batch_sz;

	static std::string parse_classname(std::string classname) {
		auto classname_ns = std::find(classname.rbegin(), classname.rend(), ':');
		classname.erase(classname.rend().base(), classname_ns.base());
		return classname;
	}

public:

	static constexpr bool default_learning_rate = .01;

	/**
	 * m_classname should be initialized by supplying `__func__` to the first argument
	 * of the Layer_Base. `parse_classname()` will normalize the string
	 * as `__func__` is compiler dependent.
	 */
	Layer_Base(std::string classname_, BC::size_t inputs, BC::size_t outputs):
		m_classname(parse_classname(classname_)),
		m_input_sz(inputs),
		m_output_sz(outputs),
		m_batch_sz(1) {}

	void resize(BC::size_t inputs, BC::size_t outputs) {
		m_input_sz = inputs;
		m_output_sz = outputs;
	}

	std::string classname() const { return m_classname; }
	std::string get_string_architecture() const {
		std::string yaml_str =  classname() + ':'
				+ "\n\tinputs: " + std::to_string(input_size())
				+ "\n\toutputs: " + std::to_string(output_size());

		if (m_additional_architecture_features != "") {
			yaml_str += "\n\t" + m_additional_architecture_features;
		}

		return yaml_str;
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

	BC::size_t  input_size() const { return m_input_sz; }
	BC::size_t  output_size() const { return m_output_sz; }
	BC::size_t  batch_size()   const { return m_batch_sz; }

	BC::size_t  batched_input_size() const { return m_input_sz * m_batch_sz; }
	BC::size_t  batched_output_size() const { return m_output_sz * m_batch_sz; }

	void set_batch_size(int bs) { m_batch_sz = bs;}
	void set_learning_rate(int) {}
	void update_weights() {}

	void save(Layer_Loader&) {};
	void load(Layer_Loader&) {};


};

}
}



#endif /* LAYER_H_ */
