
#ifndef BLACKCAT_TUPLE
#define BLACKCAT_TUPLE

#include "Layers/InputLayer.h"
#include "Layers/OutputLayer.h"
namespace BC {
namespace NN {

template<int index, class derived, class...>
struct LayerChain;

//TAIL
template<int index, class derived, class output_layer_t>
struct LayerChain<index, derived, output_layer_t> {

    using self = LayerChain<index, derived, output_layer_t>;
    using type = output_layer_t;

    output_layer_t layer;

    LayerChain(int x) : layer(x) {}
    const auto& head() const { return prev().head(); }
          auto& head()       { return prev().head(); }
    const auto& tail() const { return *this; }
          auto& tail()          { return *this; }
    const auto& prev() const { return static_cast<const derived&>(*this); }
          auto& prev()         { return static_cast<        derived&>(*this); }

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

    template<class param, class... integers>
    LayerChain(param i, param o, integers... dims) :  parent(o, dims...), layer(i,o) {}

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

    vec outputs;
    vec deltas;

    int batch_size = 1;

    template<class... integers>
    Chain(int x, integers... dims) : parent(x, x, dims...) {} //first layer is always input layer (so we double x)

    template<class T> const auto& back_propagation(const T& tensor_expected) { return this->tail().bp(tensor_expected); }
    template<class T> const auto& forward_propagation(const T& tensor_expected) { return this->tail().bp(tensor_expected); }

    void read(std::ifstream& is)         { this->for_each([&](auto& layer) { layer.read(is);     });}
    void write(std::ifstream& os)         { this->for_each([&](auto& layer) { layer.write(os);     });}
    void set_batch_size(int x)             { this->for_each([&](auto& layer) { layer.set_batch_size(x);    });}
    void update_weights()                 { this->for_each([ ](auto& layer) { layer.update_weights();        });}
    void cache_gradients()                { this->for_each([ ](auto& layer) { layer.cache_gradients();    });}
    void set_max_bptt_length(int len)   { this->for_each_internal([&](auto& layer)  { layer.set_max_bptt_length(len);}); }
    void set_learning_rate(fp_type lr)    { this->for_each([&](auto& layer) { layer.set_learning_rate(lr);     });}
//    void clear_stored_gradients()        { this->for_each([ ](auto& layer) { layer.clear_stored_gradients(); }); }

    template<class function>
    void for_each_internal(function f) {
        //same as for each but excludes input and output layers
        this->next().for_each_internal(f);
    }

    void initialize_variables() {
//        initialize_workspace_variables();
//        initialize_layer_views();
    }
private:
    void initialize_workspace_variables() {
        int workspace_size  = 0;

        this->for_each_internal([&](auto& layer) {
            workspace_size  += layer.outputs().size();
        });

        outputs = vec(workspace_size);
        deltas  = vec(workspace_size);
    }
    void initialize_layer_views() {
        initialize_workspace_variables();
        int activation_offset = 0;

        this->for_each_internal([&](auto& layer) {
            int a_sz = layer.outputs().size();
            auto activation_workspace = outputs[{activation_offset, activation_offset + a_sz}];
            auto delta_workspace      = deltas[{activation_offset, activation_offset + a_sz}];
            layer.set_activation(activation_workspace, delta_workspace);
            activation_offset += layer.outputs().size();
        });
    }


    //required terminating methods
public:
    template<class T>
    const auto& bp(const T& dx) { return dx; }
    auto& head() { return this->data(); }
};
}
}
#endif
