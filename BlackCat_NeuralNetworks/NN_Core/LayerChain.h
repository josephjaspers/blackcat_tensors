
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

#include "Layers/InputLayer.h"
#include "Layers/OutputLayer.h"
namespace BC {
namespace NN {


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
		LayerChain(param x, integers... dims) :  parent(dims...), type(x) {}

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
	};

	//HEAD
	template<class chain_Base, template<class> class... lst>
	struct LayerChain<0, chain_Base, InputLayer, lst...>
		: public LayerChain<1, LayerChain<0, chain_Base, InputLayer, lst...>, lst...>,
		  public InputLayer<LayerChain<0, chain_Base, InputLayer, lst...>> {

		static constexpr int index = 0;
		using self = LayerChain<index, chain_Base, InputLayer, lst...>;
		using parent = LayerChain<index + 1, self, lst...>;
		using type = InputLayer<self>;


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

	//HEAD
	template<template<class> class... lst>
	struct Chain
		: public LayerChain<0, Chain<lst...>, lst...>{

		using self = Chain<lst...>;
		using parent = LayerChain<0, self, lst...>;

		template<class... integers>
		Chain(integers... dims) : parent(dims...) {}

	};

}
}
#endif
