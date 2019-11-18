/*
 * NeuralNetwork.h
 *
 *  Created on: Mar 5, 2018
 *	  Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORK_NEURALNETWORK_H_
#define BLACKCAT_NEURALNETWORK_NEURALNETWORK_H_

#include "layers/Layer_Traits.h"

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
}

/** the Neural_Network
 *
 *
 *
 *
 */
template<class ... Layers>
struct NeuralNetwork {

	using self = NeuralNetwork<Layers...>;
	using layer_chain = LayerChain<
	BC::traits::Integer<0>,
	BC::traits::truth_type<BC::traits::any<detail::is_recurrent_layer, Layers...>::value>,
	void,
	Layers...>;

	layer_chain m_layer_chain;
	double m_learning_rate = m_layer_chain.head().layer().get_learning_rate();
	BC::size_t m_batch_size = 1;

	/**Basic Constructor for Neural Networks.
	 * Accepts a variadic parameter pack of Layer-like objects.
	 */
	NeuralNetwork(Layers... layers):
			m_layer_chain(layers...) {
	}

	/** Calls forward propagation on each of the neural_network's layers
	 * The time index will be set to zero prior to each call forward_call.
	 *
	 * Arguments:
	 *	A Tensor type with the same shape as the first layer's
	 *	return value of 'get_batched_input_shape()'
	 *
	 * Returns:
	 * 	If the neural-network terminates with an Output_Layer
	 * 	forward-propagation will return a 'shallow-copy' of the output.
	 *
	 * 	A shallow-copy will have the behavior of a Tensor class but will act
	 * 	as a handle to the underlying Tensor.
	 * 	(Passing by value will not incur a copy).
	 */
	template<class T>
	auto forward_propagation(const T& tensor) {
		auto fp_caller = [](auto& layer, const auto& X) {
			return layer.forward_propagation(X);
		};

		zero_time_index();
		return m_layer_chain.for_each_propagate(fp_caller, tensor);
	}

	/** Calls back-propagation on each of the neural_network's layers
	 * The time index will be incremented after each call to back_propagation.
	 *
	 * Returns: the error of the inputs at the current time_index.
	 */
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

	/** Returns the output of a single batch.
	 *
	 * Predict is identical to forward propagation except
	 * it does not cache the intermediate values required for
	 * back_propagation. The inputs and outputs of each layer however
	 * are cached, as they are required for most recurrent neural networks.
	 */
	template<class T>
	auto predict(const T& tensor) {
		auto fp_caller = [](auto& layer, const auto& X) {
			return layer.predict(X);
		};

		zero_time_index();
		return m_layer_chain.for_each_propagate(fp_caller, tensor);
	}

	/** Returns the output of a single input.
	 *
	 * single_predict is identical to predict except it accepts a
	 * non-batched input to forward-propagate on.
	 */
	template<class T>
	auto single_predict(const T& tensor) {
		auto fp_caller = [](auto& layer, const auto& X) {
			return layer.single_predict(X);
		};

		zero_time_index();
		return m_layer_chain.for_each_propagate(fp_caller, tensor);
	}

	///Returns a const reference to the layer specified by the given index.
	template<int X> auto& get_layer() const {
		return m_layer_chain.get(BC::traits::Integer<X>());
	}

	///Returns a reference to the layer specified by the given index.
	template<int X> auto& get_layer() {
		return m_layer_chain.get(BC::traits::Integer<X>());
	}

	/** Sets the learning for each layer in the Neural_Network.
	 *
	 * Individual layer's learning rates can be set by accessing them
	 * via 'get_layer<int index>()' and calling 'set_learning_rate'
	 */
	void set_learning_rate(double lr) {
		m_learning_rate = lr;
		m_layer_chain.for_each([&](auto& layer) {
			layer.set_learning_rate(lr);
		});
	}

	///Returns the current global learning rate.
	double get_learning_rate() const {
		return m_learning_rate;
	}

	/**Sets the batch_size of the entire Neural_Network
	 * The intermediate values are discarded when setting the batch_size.
	 */
	void set_batch_size(int batch_sz) {
		m_batch_size = batch_sz;
		m_layer_chain.for_each([&](auto& layer) {
			layer.set_batch_size(batch_sz);
		});
	}

	void copy_training_data_to_single_predict(int batch_index) {
		m_layer_chain.for_each([&](auto& layer) {
			layer.zero_time_index();
			layer.copy_training_data_to_single_predict(batch_index);
		});
	}
	void zero_time_index() {
		m_layer_chain.for_each([&](auto& layer) {
			layer.zero_time_index();
		});
	}

	///Updates the weights of each Layer based upon the current stored gradients
	void update_weights() {
		m_layer_chain.for_each([ ](auto& layer) {layer.update_weights();});
	}

	///returns the input_size of the first layer
	BC::size_t input_size() const {
		return m_layer_chain.head().layer().input_size();
	}


	///returns the output_size of the last layer
	BC::size_t output_size() const {
		return m_layer_chain.tail().layer().output_size();
	}

	///returns the batch_size of the neural network.
	BC::size_t batch_size() const {
		return m_layer_chain.head().layer().batch_size();
	}

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
		zero_time_index();
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
			os << this->batch_size() << '\n';
		}

		//Initialize a layer loader object to load each layer
		Layer_Loader loader(directory_name);

		int index = 0;
		m_layer_chain.for_each([&](auto& layer) {
			loader.set_current_layer_index(index);
			loader.set_current_layer_name(layer.classname());
			loader.make_current_directory();
			layer.save(loader);
			index++;
		});
	}

	/** Loads a neural network from a previously saved instance.
	 *  Load expects the neural-network to have been unused in the previous state.
	 */
	void load(std::string directory_name) {
		zero_time_index();

		auto get_filepath = [&](std::string filename) {
			return directory_name + bc_directory_separator() + filename;
		};

		Layer_Loader loader(directory_name);
		int index = 0;

		m_layer_chain.for_each([&](auto& layer) {
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


/**Factory method for creating neural_networks.
 * Each layer defines its own respective factory_methods.
 * It is generally recommended to use these factory methods opposed to
 * instantiating a layer object manually.
 */
template<class ... Layers>
auto neuralnetwork(Layers ... layers) {
	return NeuralNetwork<Layers...>(layers...);
}

}
}

#endif /* NEURALNETWORK_H_ */
