/*
 * LayerLinker.h
 *
 *  Created on: Sep 15, 2018
 *      Author: joseph
 */

#ifndef DEPRECATED_LAYERLINKER_H_
#define DEPRECATED_LAYERLINKER_H_
#include <type_traits>
#include <forward_list>

namespace BC {
namespace NN {

template<class base, int index, template<class> class... layers>
class LayerLinker {

	auto& as_base()  { return static_cast<base&>(*this); }
	auto& prev() { return this->as_base().get<index - 1>().data(); }
	auto& tail() { return prev(); }

};

template<class base, int index, template<class> class layer_t, template<class> class... layer_ts>
class LayerLinker<base, index, layer_t, layer_ts...>
: public LayerLinker<base, index + 1, layer_ts...>,
  public layer_t<LayerLinker<base, index, layer_t, layer_ts...>> {


	using parent = LayerLinker<base, index + 1, layer_ts...>;
	using self   = LayerLinker<base,index, layer_t, layer_ts...>;
	using type   = layer_t<self>;

	static constexpr bool is_base = std::is_same<base, void>::value;

	auto& as_base()  { return static_cast<base&>(*this); }

 public:

	template<class... ps>
	LayerLinker(int inputs, int outputs, ps... params) : parent(outputs, params...), type(inputs, outputs){}

	template<int index_p>
	std::enable_if_t<index_p == index, self&> get() { return *this; }

	auto& data() { return static_cast<type&>(*this); }
	auto& next() { return this->as_base().get<index + 1>().data(); }
	auto& prev() { return this->as_base().get<index - 1>().data(); }
	auto& tail() { return next().tail(); }
};

namespace util {
	int sum(int x) { return x; }

	template<class... integers>
	int sum(int x, integers... ints) {
		return x + sum(ints...);
	}
}

template<template<class> class... layers>
class Linker : public LayerLinker<Linker<layers...>, 0, layers...> {

	using parent = LayerLinker<Linker<layers...>, 0, layers...>;

	int workspace_size;
	int batch_size = 1;

//	std::forward_list<vec> past_workspaces;
	vec workspace;
public:
	auto& head() { return static_cast<parent&>(*this).data(); }
	auto& tail() { return head().tail(); }

	template<class... params>
	Linker(params... ps) : parent(ps...), workspace_size(util::sum(ps...)) {
		head().init_input_view(workspace, 0);
	}

	void set_batch_size(int x) {
		workspace = vec(x * workspace_size);
		batch_size = x;
	}
};


}
}

#endif /* DEPRECATED_LAYERLINKER_H_ */
