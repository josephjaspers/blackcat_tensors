
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

namespace BC {
namespace nn {

template<int index, class Derived, class...>
struct LayerChain;

//TAIL
template<int index, class Derived, class OutputLayer>
struct LayerChain<index, Derived, OutputLayer> {

    using type = OutputLayer;

    OutputLayer layer;

    template<class... Args>
    LayerChain(Args... args_) : layer(args_...) {}
    const auto& head() const { return prev().head(); }
          auto& head()       { return prev().head(); }
    const auto& tail() const { return *this; }
          auto& tail()       { return *this; }
    const auto& prev() const { return static_cast<const Derived&>(*this); }
          auto& prev()       { return static_cast<      Derived&>(*this); }

    template<class T> const auto& fp(const T& tensor) { return layer.forward_propagation(tensor); }
    template<class T> const auto bp(const T& tensor)  { return this->prev().bp(layer.back_propagation(tensor)); }
    template<class function> void for_each(function f) { f(layer); }
};

//BODY
template<int index, class Derived,class CurrentLayer, class... Layers>
struct LayerChain<index, Derived, CurrentLayer, Layers...>
: LayerChain<index + 1, LayerChain<index, Derived, CurrentLayer, Layers...>, Layers...> {

    using self   = LayerChain<index, Derived, CurrentLayer, Layers...>;
    using parent = LayerChain<index + 1, self, Layers...>;
    using type   = CurrentLayer;

    CurrentLayer layer;

    LayerChain(CurrentLayer f, Layers... layers):
    	parent(layers...),
    	layer(f) {}

    const auto& head() const { return prev().head(); }
          auto& head()       { return prev().head(); }
    const auto& tail() const { return next().tail(); }
          auto& tail()       { return next().tail(); }
    const auto& next() const { return static_cast<const parent&>(*this); }
          auto& next()       { return static_cast<        parent&>(*this); }
    const auto& prev() const { return static_cast<const Derived&>(*this); }
          auto& prev()       { return static_cast<        Derived&>(*this); }

    template<class T> const auto& fp(const T& tensor) { return this->next().fp(layer.forward_propagation(tensor)); }
    template<class T> const auto bp(const T& tensor) { return this->prev().bp(layer.back_propagation(tensor)); }

    template<class function> void for_each(function f) {
        f(layer);
        next().for_each(f);
    }
};

//HEAD
template<class Derived,class CurrentLayer, class... Layers>
struct LayerChain<0, Derived, CurrentLayer, Layers...>
: LayerChain<1, LayerChain<0, Derived, CurrentLayer, Layers...>, Layers...> {

	static_assert(std::is_void<Derived>::value, "First Derived class must be void of LayerChain");
    using self   = LayerChain<0, Derived, CurrentLayer, Layers...>;
    using parent = LayerChain<1, self, Layers...>;
    using type   = CurrentLayer;

    CurrentLayer layer;

    LayerChain(CurrentLayer f, Layers... layers):
    	parent(layers...),
    	layer(f) {}

    const auto& head() const { return *this; }
          auto& head()       { return *this; }
    const auto& tail() const { return next().tail(); }
          auto& tail()       { return next().tail(); }
    const auto& next() const { return static_cast<const parent&>(*this); }
          auto& next()       { return static_cast<      parent&>(*this); }
    const auto& prev() const { return *this; }
          auto& prev()       { return *this; }

    template<class T> const auto& fp(const T& tensor) { return this->next().fp(layer.forward_propagation(tensor)); }
    template<class T> const auto bp(const T& tensor) { return layer.back_propagation(tensor); }

    template<class function> void for_each(function f) {
        f(layer);
        next().for_each(f);
    }

};

//HEAD
template<class... Layers>
struct Chain:
		public LayerChain<0, void, Layers...>{

    using self = Chain<Layers...>;
    using parent = LayerChain<0, void, Layers...>;

    Chain(Layers... layers):
    	parent(layers...) {}

    template<class T> const auto back_propagation(const T& tensor_expected) { return this->tail().bp(tensor_expected); }
    template<class T> const auto& forward_propagation(const T& tensor_expected) { return this->head().fp(tensor_expected); }

    void read(std::ifstream& is)         { this->for_each([&](auto& layer) { layer.read(is);     });}
    void write(std::ifstream& os)         { this->for_each([&](auto& layer) { layer.write(os);     });}
    void set_batch_size(int x)             { this->for_each([&](auto& layer) { layer.set_batch_size(x);    });}

    void update_weights()                 { this->for_each([ ](auto& layer) { layer.update_weights();        });}
    void set_max_bptt_length(int len)   { this->for_each([&](auto& layer)  { layer.set_max_bptt_length(len);}); }
};
}
}
#endif
