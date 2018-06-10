
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

#include "Defaults.h"
#include "Layers/InputLayer.h"
#include "Layers/OutputLayer.h"
#include "Layers/Layer.h"
namespace BC {
namespace NN {
template<class derived, template<class> class...> struct LayerChain;
template<class> class OutputLayer;
template<class> class InputLayer;

	//Layer chain - a tuple that inherits from each of its types and has an data two iteration technique
	//this is a bidirectional tuple

	template<class derived>
	struct LayerChain<derived, OutputLayer> : public OutputLayer<LayerChain<derived, OutputLayer>>{

		using p = derived;
		using me = LayerChain<derived, OutputLayer>;
		using type = OutputLayer<LayerChain<derived, OutputLayer>>;
		using next_type = void;

		LayerChain(int x) : type(x) {}
		bool hasNext() const { return false; }

		const auto& tail() const { return *this; }
			  auto& tail() 		 { return *this; }

		const auto& head() const { return prev().head(); }
			  auto& head()  	 { return prev().head(); }

		const me& next() const { throw std::invalid_argument("no next end of chain"); return *this;}
			  me& next() 		 { throw std::invalid_argument("no next end of chain"); return *this;}

		const auto& prev() const { return static_cast<p&>(*this).data(); }
			  auto& prev() 		 { return static_cast<p&>(*this).data(); }


		const auto& data() const { return static_cast<const type&>(*this); }
			  auto& data()  	 { return static_cast<		type&>(*this); }
	};

	//BODY
	template<class derived, template<class> class front, template<class> class... lst>
	struct LayerChain<derived, front, lst...> : LayerChain<LayerChain<derived, front, lst...>, lst...>,
												public front<LayerChain<derived, front, lst...>> {

		using p = derived;
		using n = typename LayerChain<LayerChain<derived, front, lst...>, lst...>::type;
		using parent = LayerChain<LayerChain<derived, front, lst...>, lst...>;
		using me = LayerChain<derived, front, lst...>;
		using type = front<LayerChain<derived, front, lst...>>;
		using next_type = n;

		bool hasNext() const { return true; }

		template<class param, class... integers>
		LayerChain(param x, integers... dims) : type(x), parent(dims...) {}

		const auto& tail() const { return next().tail(); }
		const auto& head() const { return prev().head(); }

		auto& tail() { return static_cast<parent&>(*this).tail(); }
		auto& head() { return prev().head(); }

			  auto& prev()  	 { return static_cast<p&>(*this).data(); }
		const auto& prev() const { return static_cast<p&>(*this).data(); }

			  auto& next()    	 { return static_cast<parent&>(*this).data(); }
		const auto& next() const { return static_cast<parent&>(*this).data(); }

		const auto& data() const { return static_cast<const type&>(*this); }
		 	  auto& data()  	 { return static_cast<		type&>(*this); }
	};

	//HEAD
	template<template<class> class... lst>
	struct LayerChain<BASE, InputLayer, lst...>
		: public LayerChain<LayerChain<BASE, InputLayer, lst...>, lst...>,
		  public InputLayer<LayerChain<BASE, InputLayer, lst...>> {

		using n = typename LayerChain<LayerChain<BASE, InputLayer, lst...>, lst...>::type;
		using parent = LayerChain<LayerChain<BASE, InputLayer, lst...>, lst...>;
		using me = LayerChain<BASE, InputLayer, lst...>;
		using type = InputLayer<LayerChain<BASE, InputLayer, lst...>>;
		using next_type = n;


		template<class param, class... integers>
		LayerChain(param x, integers... dims) : parent(x, dims...) {}

		bool hasNext() const { return true; }
		const auto& data() const { return static_cast<const type&>(*this); }
			  auto& data()  	 { return static_cast<		type&>(*this); }

		const auto& tail() const { return next().tail(); }
			  auto& tail() 		 { return static_cast<parent&>(*this).tail(); }
		const auto& head() const { return data(); }
			  auto& head()  	 { return data(); }
		const auto& next() const { return static_cast<parent&>(*this).data(); }
			  auto& next()  	 { return static_cast<parent&>(*this).data(); }
	};

}
}
#endif
