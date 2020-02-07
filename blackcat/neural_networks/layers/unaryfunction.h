
#ifndef BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_
#define BLACKCAT_NEURALNETWORK_UNARYFUNCTION_H_

#include "layer_base.h"
#include <string>

namespace bc {
namespace nn {

template<
	class Functor,
	class Dimension,
	class ValueType,
	class SystemTag,
	class Allocator=layer_default_allocator<SystemTag, ValueType>,
	class OutputDimension=Dimension,
	class OutputValueType=ValueType,
	class OutputSystemTag=SystemTag,
	class OutputAllocator=Allocator>
struct Function:
		Layer_Base<
			Dimension,
			ValueType,
			SystemTag,
			Allocator,
			OutputDimension,
			OutputValueType,
			OutputSystemTag,
			OutputAllocator>
{
	using parent_type = Layer_Base<
			Dimension,
			ValueType,
			SystemTag,
			Allocator,
			OutputDimension,
			OutputValueType,
			OutputSystemTag,
			OutputAllocator>;

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using self_type = Function<SystemTag, ValueType, Functor,InputDimension>;

	using input_tensor_dim = Dimension;
	using output_tensor_dim = OutputDimension;
	using shape_type = bc::Dim<input_tensor_dim::value>;

	Functor function;
	shape_type m_input_shape;

	Function(shape_type inputs, Functor func=Functor()):
		parent_type(bc_get_classname_of(func)),
		function(func) {}

	void init()
	{
		this->m_inner_shape = this->prev().shape();
		this->m_output_shape = this->next().shape();
	}

	virtual batched_output_tensor_type forward_propagation (
			const batched_input_tensor_type& x) override
	{
		return function(x);
	}

	virtual batched_input_tensor_type back_propagation(
			const batched_output_tensor_type& dy) override
	{
		return function.cached_dx(this->prev().y);
	}

	virtual void set_learning_rate_hook(double lr) override
	{
		w_opt.set_learning_rate(this->batched_learning_rate());
		b_opt.set_learning_rate(this->batched_learning_rate());
	}

	void update_weights()
	{
		w_opt.update(w, w_gradients);
		b_opt.update(b, b_gradients);
		w_gradients.zero();
		b_gradients.zero();
	}

	virtual void save(Layer_Loader& loader) const override
	{
		loader.save_variable(w, "w");
		loader.save_variable(b, "b");
		w_opt.save(loader, "w_opt");
		b_opt.save(loader, "b_opt");
	}

	virtual void load(Layer_Loader& loader) override
	{
		loader.load_variable(w, "w");
		loader.load_variable(b, "b");
		w_opt.load(loader, "w_opt");
		b_opt.save(loader, "b_opt");
	}
};


template<class ValueType, class SystemTag, class Functor>
Function<SystemTag, ValueType, Functor> function(SystemTag system_tag, int inputs, Functor function=Functor()) {
	return Function<SystemTag, ValueType, Functor>(bc::Dim<1>{inputs}, function);
}

template<class SystemTag, class Functor>
auto function(SystemTag system_tag, int inputs, Functor function=Functor()) {
	return Function<SystemTag, typename SystemTag::default_floating_point_type, Functor>(bc::Dim<1>{inputs}, function);
}


template<class ValueType, class SystemTag, class Functor, int X>
Function<SystemTag, ValueType, Functor> function(SystemTag system_tag, Dim<X> shape, Functor function=Functor()) {
	return Function<SystemTag, ValueType, Functor, bc::traits::Integer<X>>(shape, function);
}

template<class SystemTag, class Functor, int X>
auto function(SystemTag system_tag, bc::Dim<X> shape, Functor function=Functor()) {
	return Function<SystemTag, typename SystemTag::default_floating_point_type, Functor, bc::traits::Integer<X>>(shape, function);
}


}
}


#endif 
