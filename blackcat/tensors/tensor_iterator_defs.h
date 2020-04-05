
#define BC_FORWARD_ITER(suffix, iter, access)                      \
	auto suffix##iter() const {                                \
		return iterators::iter_##suffix##iter(access);     \
	}                                                          \
	auto suffix##iter() {                                      \
		return iterators::iter_##suffix##iter(access);     \
	}                                                          \
	auto suffix##c##iter() const {                             \
		return iterators::iter_##suffix##iter(access);     \
	}                                                          \
	auto suffix##r##iter() const {                             \
		return iterators::iter_##suffix##r##iter(access);  \
	}                                                          \
	auto suffix##r##iter() {                                   \
		return iterators::iter_##suffix##r##iter(access);  \
	}                                                          \
	auto suffix##cr##iter() const {                            \
		return iterators::iter_##suffix##r##iter(access);  \
	}

#define BC_ITERATOR_DEF(suffix, iterator_name, begin_func, end_func)     \
                                                                         \
    template<class Tensor>                                               \
    struct iterator_name {                                               \
                                                                         \
        using size_t = bc::size_t;                                       \
        Tensor& tensor;                                                  \
                                                                         \
        using begin_t = decltype(tensor.begin_func ());                  \
        using end_t = decltype(tensor.end_func ());                      \
                                                                         \
        begin_t m_begin = tensor.begin_func();                           \
        end_t m_end = tensor.end_func();                                 \
                                                                         \
        iterator_name(Tensor& tensor) :                                  \
                tensor(tensor) {}                                        \
                                                                         \
        iterator_name(Tensor& tensor, size_t start):                     \
            tensor(tensor)                                               \
        {                                                                \
            m_begin += start;                                            \
        }                                                                \
                                                                         \
        iterator_name(Tensor& tensor, size_t start, size_t end):         \
                tensor(tensor)                                           \
        {                                                                \
            m_begin += start;                                            \
            m_end = end;                                                 \
        }                                                                \
                                                                         \
        auto begin() {                                                   \
            return m_begin;                                              \
        }                                                                \
                                                                         \
        const begin_t& cbegin() const {                                  \
            return m_begin;                                              \
        }                                                                \
                                                                         \
        const end_t& end() const {                                       \
            return m_end;                                                \
        }                                                                \
    };                                                                   \
                                                                         \
                                                                         \
template<class... params> auto suffix##iter(params... ps) const {        \
    return iterator_name<const self_type>(*this, ps...);                 \
}                                                                        \
                                                                         \
template<class... params> auto suffix##iter(params... ps) {              \
    return iterator_name<self_type>(*this, ps...);                       \
}                                                                        \
                                                                         \
template<class... params> auto suffix##const_iter(params... ps) const {  \
    return iterator_name<const self_type>(*this, ps...);                 \
}

