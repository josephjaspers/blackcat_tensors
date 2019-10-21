/*
 * NeuralNetwork.h
 *
 *  Created on: Mar 5, 2018
 *	  Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORK_NEURALNETWORK_H_
#define BLACKCAT_NEURALNETWORK_NEURALNETWORK_H_

#include "Layers/Layer_Traits.h"

#include "Layer_Chain.h"
#include "Layer_Loader.h"
#include <sys/types.h>
#include <sys/stat.h>

namespace BC {
namespace nn {
namespace detail {

template<class T>
using is_recurrent_layer = BC::traits::conditional_detected_t<
			detail::query_forward_requires_outputs, T, std::false_type>;

template<class... Layers>
struct NeuralNetwork {

	using self = NeuralNetwork<Layers...>;
	using layer_chain = LayerChain<
			BC::traits::Integer<0>,
			BC::traits::truth_type<BC::traits::any<detail::is_recurrent_layer, Layers...>::value>,
			void,
			Layers...>;

	layer_chain m_layer_chain;
	double m_learning_rate = Layer_Base::default_learning_rate;
	BC::size_t m_batch_size = 1;

	NeuralNetwork(Layers... layers):
		m_layer_chain(layers...) {}

	template<class T>
	auto forward_propagation(const T& tensor) {
		auto fp_caller = [](auto& layer, const auto& X) {
			return layer.forward_propagation(X);
		};

		m_layer_chain.for_each([&](auto& layer) {
			layer.zero_time_index();
		});

		return m_layer_chain.for_each_propagate(fp_caller, tensor);
	}

	template<class T>
	auto back_propagation(const T& tensor) {
		auto bp_caller = [](auto& layer, const auto& Dy) {
			return layer.back_propagation(Dy);
		};

		auto& last_layer = m_layer_chain.tail();
		auto dx = last_layer.reverse_for_each_propagate(bp_caller, tensor);

		m_layer_chain.for_each([&](auto& layer) {
			layer.increment_time_index();
		});

		return dx;
	}

	template<class T>
	auto predict(const T& tensor) {
		m_layer_chain.for_each([&](auto& layer) {
			layer.zero_time_index();
		});

		auto fp_caller = [](auto& layer, const auto& X) {
			return layer.predict(X);
		};

		return m_layer_chain.for_each_propagate(fp_caller, tensor);
	}

	template<class T>
	auto single_predict(const T& tensor) {
		m_layer_chain.for_each([&](auto& layer) {
			layer.zero_time_index();
		});

		auto fp_caller = [](auto& layer, const auto& X) {
			return layer.single_predict(X);
		};

		return m_layer_chain.for_each_propagate(fp_caller, tensor);
	}

	template<int X> auto& get_layer() const { return m_layer_chain.get(BC::traits::Integer<X>()); }
	template<int X> auto& get_layer() { return m_layer_chain.get(BC::traits::Integer<X>()); }

	void set_learning_rate(double lr) { m_learning_rate = lr; m_layer_chain.for_each([&](auto& layer) { layer.set_learning_rate(lr); }); }
	double get_learning_rate() const { return m_learning_rate; }

	void set_batch_size(int x) { m_batch_size = x; m_layer_chain.for_each([&](auto& layer) { layer.set_batch_size(x);	});}
	BC::size_t get_batch_size() const { return m_batch_size; }
	void update_weights()	  { m_layer_chain.for_each([ ](auto& layer) { layer.update_weights(); });}

	BC::size_t input_size() const { return m_layer_chain.head().layer().input_size(); }
	BC::size_t output_size() const { return m_layer_chain.tail().layer().output_size(); }
	BC::size_t batch_size() const { return m_layer_chain.head().layer().batch_size(); }

	/**
	 * Returns a yaml representation of the neural network
	 */
	std::string get_string_architecture() const {
		std::string architecture = "";
		m_layer_chain.for_each([&](auto& layer) {
			architecture += layer.get_string_architecture() + "\n";
		});
		return architecture;
	}

	/**
	 * Creates the directory `directory_name` using mkdir.
	 * Than outputs an architecture.yaml file with a description of the neural network.
	 * Than iterates through each layer and calls 'save(int layer_index, std::string directory_name)'
	 */

	void save(std::string directory_name) {
		//Attempt to create a directory to save our model in
		if ((directory_name != "") && (directory_name != ".")) {
			int error = system(std::string("mkdir " + directory_name).c_str());
		}

		auto get_filepath = [&](std::string filename) {
			return directory_name + bc_directory_separator() + filename;
		};

		{
			//Create a yaml file with network description/architecture
			std::ofstream os(get_filepath("architecture.yaml"));
			os << get_string_architecture();
		}

		{
			//Store meta-data about the current state
			std::ofstream os(get_filepath("meta"));
			os << this->get_learning_rate() << '\n';
			os << this->get_batch_size() << '\n';
		}

		//Initialize a layer loader object to load each layer
		Layer_Loader loader(directory_name);

		int index = 0;
		m_layer_chain.for_each([&](auto& layer){
			loader.set_current_layer_index(index);
			loader.set_current_layer_name(layer.classname());
			loader.make_current_directory();
			layer.save(loader);
			index++;
		});
	}

	void load(std::string directory_name) {

		auto get_filepath = [&](std::string filename) {
			return directory_name + bc_directory_separator() + filename;
		};

		Layer_Loader loader(directory_name);
		int index = 0;
		m_layer_chain.for_each([&](auto& layer){
			loader.set_current_layer_index(index);
			loader.set_current_layer_name(layer.classname());
			layer.load(loader);
			index++;
		});

		std::ifstream is(get_filepath("meta"));
		std::string tmp;

		std::getline(is, tmp);
		set_learning_rate(std::stod(tmp));
		std::getline(is, tmp);
		set_batch_size(std::stoi(tmp));
	}
};

template<class... Layers>
auto neuralnetwork(Layers... layers) {
	return NeuralNetwork<Layers...>(layers...);
}

}
}

#endif /* NEURALNETWORK_H_ */
