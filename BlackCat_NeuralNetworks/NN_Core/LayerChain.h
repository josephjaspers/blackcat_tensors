
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

#include "Layers/InputLayer.h"
#include "Layers/OutputLayer.h"
namespace BC {
namespace NN {

//Basically a specialized tuple

template<int index, class derived, template<class> class...> struct LayerChain;
template<class> class OutputLayer;
template<class> class InputLayer;

	//TAIL
	template<int index, class derived>
	struct LayerChain<index, derived, OutputLayer> : public OutputLayer<LayerChain<index, derived, OutputLayer>>{

		using self = LayerChain<index, derived, OutputLayer>;
		using type = OutputLayer<self>;

		LayerChain(int x) : type(x) {}
		bool hasNext() const { return false; }

		const auto& tail() const { return *this; }
			  auto& tail() 		 { return *this; }

		const auto& head() const { return prev().head(); }
			  auto& head()  	 { return prev().head(); }

		const self& next() const { throw std::invalid_argument("no next end of chain");}
			  self& next() 		 { throw std::invalid_argument("no next end of chain");}

		const auto& prev() const { return static_cast<const derived&>(*this).data(); }
			  auto& prev() 		 { return static_cast<derived&>(*this).data(); }

		const auto& data() const { return static_cast<const type&>(*this); }
			  auto& data()  	 { return static_cast<		type&>(*this); }

		  	  auto& prev_link()    	 { return static_cast<derived&>(*this); }
		  const auto& prev_link() const { return static_cast<const derived&>(*this); }

		template<class T>
		const auto& fp(const T& tensor) { return this->data().forward_propagation(tensor); }
		template<class T> const auto& bp(const T& tensor) { return this->prev_link().bp(data().back_propagation(tensor)); }

		template<class function> void for_each(function f) {f(data()); };

	};

	//BODY
	template<int index, class derived, template<class> class front, template<class> class... lst>
	struct LayerChain<index, derived, front, lst...> : LayerChain<index + 1, LayerChain<index, derived, front, lst...>, lst...>,
												public front<LayerChain<index, derived, front, lst...>> {

		using self = LayerChain<index, derived, front, lst...>;
		using parent = LayerChain<index + 1, self, lst...>;
		using type = front<self>;

		bool hasNext() const { return true; }

		template<class param, class... integers>
		LayerChain(param i, param o, integers... dims) :  parent(o, dims...), type(i,o) {}

		const auto& tail() const { return next().tail(); }
		const auto& head() const { return prev().head(); }

		auto& tail() { return static_cast<parent&>(*this).tail(); }
		auto& head() { return prev().head(); }

			  auto& prev()  	 { return static_cast<derived&>(*this).data(); }
		const auto& prev() const { return static_cast<const derived&>(*this).data(); }

			  auto& next()    	 { return static_cast<parent&>(*this).data(); }
		const auto& next() const { return static_cast<const parent&>(*this).data(); }


		const auto& data() const { return static_cast<const type&>(*this); }
		 	  auto& data()  	 { return static_cast<		type&>(*this); }

			  auto& next_link()    	 { return static_cast<parent&>(*this); }
		const auto& next_link() const { return static_cast<const parent&>(*this); }
		  	  auto& prev_link()    	 { return static_cast<derived&>(*this); }
		  const auto& prev_link() const { return static_cast<const derived&>(*this); }

		template<class T> const auto& fp(const T& tensor) { return this->next_link().fp(data().forward_propagation(tensor)); }
		template<class T> const auto& bp(const T& tensor) { return this->prev_link().bp(data().back_propagation(tensor)); }
		template<class function> void for_each(function f) { f(data()); next_link().for_each(f); };
	};

	//HEAD
	template<template<class> class... lst>
	struct Chain
		: public LayerChain<0, Chain<lst...>, lst...>{

		using self = Chain<lst...>;
		using parent = LayerChain<0, self, lst...>;

		template<class... integers>
		Chain(int x, integers... dims) : parent(x, x, dims...) {} //first layer is always input layer


		template<class T>
		const auto& backprop(const T& tensor_expected) { return this->tail().bp(tensor_expected); }

		void read(std::ifstream& is) { this->for_each([&](auto& layer) { layer.read(is); });}
		void write(std::ifstream& os) { this->for_each([&](auto& layer) { layer.write(os); });}
		void set_batch_size(int x) { this->for_each([&](auto& layer) { layer.set_batch_size(x); }); }
		void clear_stored_delta_gradients() { this->for_each([](auto& layer) { layer.clear_stored_delta_gradients(); }); }
		void update_weights() { this->for_each([](auto& layer) { layer.update_weights(); }); }
		void setLearningRate(fp_type lr) { this->for_each([&](auto& layer) { layer.setLearningRate(lr); }); }


		template<class T>
		const auto& bp(const T& dx) { return dx; }
		auto& head() { return this->data(); }


	};
}
}
#endif
