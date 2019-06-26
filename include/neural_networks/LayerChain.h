
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

namespace BC {
namespace nn {

template<int index, class derived, class...>
struct LayerChain;

//TAIL
template<int index, class derived, class output_layer_t>
struct LayerChain<index, derived, output_layer_t> {

    using self = LayerChain<index, derived, output_layer_t>;
    using type = output_layer_t;

    output_layer_t layer;

    template<class... Args>
    LayerChain(Args... args_) : layer(args_...) {}
    const auto& head() const { return prev().head(); }
          auto& head()       { return prev().head(); }
    const auto& tail() const { return *this; }
          auto& tail()          { return *this; }
    const auto& prev() const { return static_cast<const derived&>(*this); }
          auto& prev()       { return static_cast<      derived&>(*this); }

    template<class T> const auto& fp(const T& tensor) { return layer.forward_propagation(tensor); }
    template<class T> const auto& bp(const T& tensor) { return this->prev().bp(layer.back_propagation(tensor)); }
    template<class function> void for_each(function f) { f(layer); }
    template<class function> void for_each_internal(function f) {}

};

//BODY
template<int index, class derived,class front, class... lst>
struct LayerChain<index, derived, front, lst...>
: LayerChain<index + 1, LayerChain<index, derived, front, lst...>, lst...> {

    using self         = LayerChain<index, derived, front, lst...>;
    using parent     = LayerChain<index + 1, self, lst...>;
    using type         = front;

    front layer;

    template<class... integers>
    LayerChain(size_t i, size_t o, integers... dims) :  parent(o, dims...), layer(i,o) {}

    template<class... integers>
    LayerChain(front f, integers... dims) :  parent(dims...), layer(f) {}


    const auto& head() const { return prev().head(); }
          auto& head()          { return prev().head(); }
    const auto& tail() const { return next().tail(); }
          auto& tail()          { return next().tail(); }
    const auto& next() const { return static_cast<const parent&>(*this); }
          auto& next()         { return static_cast<        parent&>(*this); }
    const auto& prev() const { return static_cast<const derived&>(*this); }
          auto& prev()         { return static_cast<        derived&>(*this); }

    template<class T> const auto& fp(const T& tensor) { return this->next().fp(layer.forward_propagation(tensor)); }
    template<class T> const auto& bp(const T& tensor) { return this->prev().bp(layer.back_propagation(tensor)); }

    template<class function> void for_each(function f) {
        f(layer);
        next().for_each(f);
    }
    template<class function> void for_each_internal(function f) {
        f(layer);
        next().for_each_internal(f);
    }
};

//HEAD
template<class... lst>
struct Chain : public LayerChain<0, Chain<lst...>, lst...>{

    using self = Chain<lst...>;
    using parent = LayerChain<0, self, lst...>;

    BC::size_t  batch_size = 1;

    template<class... Args>
    Chain(Args&&... args) : parent(args...) {} //first layer is always input layer (so we double x)

    template<class T> const auto& back_propagation(const T& tensor_expected) { return this->tail().bp(tensor_expected); }
    template<class T> const auto& forward_propagation(const T& tensor_expected) { return this->head().fp(tensor_expected); }

    void read(std::ifstream& is)         { this->for_each([&](auto& layer) { layer.read(is);     });}
    void write(std::ifstream& os)         { this->for_each([&](auto& layer) { layer.write(os);     });}
    void set_batch_size(int x)             { this->for_each([&](auto& layer) { layer.set_batch_size(x);    });}

    template<class SystemTag>
    void initialize_variables(BC::Stream<SystemTag> stream)             { this->for_each([&](auto& layer) { layer.initialize_variables(stream);    });}

    template<class SystemTag>
    size_t get_workspace_size(BC::Stream<SystemTag> stream){
    	size_t size;
    	this->for_each([&](auto& layer) { size += layer.get_workspace_size(stream);    });
    }


    void update_weights()                 { this->for_each([ ](auto& layer) { layer.update_weights();        });}
    void cache_gradients()                { this->for_each([ ](auto& layer) { layer.cache_gradients();    });}
    void set_max_bptt_length(int len)   { this->for_each_internal([&](auto& layer)  { layer.set_max_bptt_length(len);}); }
//    void clear_stored_gradients()        { this->for_each([ ](auto& layer) { layer.clear_stored_gradients(); }); }

    template<class function>
    void for_each_internal(function f) {
        //same as for each but excludes input and output layers
        this->next().for_each_internal(f);
    }

public:
    template<class T>
    const auto& bp(const T& dx) { return dx; }
    auto& head() { return this->data(); }
};
}
}
#endif
