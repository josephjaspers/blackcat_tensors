#ifndef BLACKCAT_TENSORS_EXPRESSION_TEMPLATES_EXPRESSION_MASK_H_
#define BLACKCAT_TENSORS_EXPRESSION_TEMPLATES_EXPRESSION_MASK_H_

#include "expression_template_base.h"
#include "expression_binary.h"

namespace bc {
namespace tensors {
namespace exprs {

struct Mask {};

template<class Lv, class MaskArray>
struct Bin_Op<Mask, Lv, MaskArray>: Expression_Base<Bin_Op<Mask, Lv, MaskArray>>
{
	using system_tag = typename Lv::system_tag;
	using value_type = typename Lv::value_type;

	static constexpr int tensor_dim = Lv::tensor_dim;
	static constexpr int tensor_iterator_dim = bc::traits::max(
		tensor_dim, Lv::tensor_iterator_dim, MaskArray::tensor_iterator_dim);

	static_assert(Lv::tensor_dim == MaskArray::tensor_dim,
	"Mask must have same dimensions");

public:

	Lv left;
	MaskArray mask;

	template<class... Args> BCHOT
	Bin_Op(Lv lv, MaskArray rv, const Args&... args):
		left(lv),
		mask(rv) {}

	BCINLINE
	void operator [](bc::size_t index) const {
		if (mask[index])
			left[index];
	}

	BCINLINE
	void operator [](bc::size_t index) {
		if (mask[index])
			left[index];
	}

	template<
		class... Integers,
		class=std::enable_if_t<
				(sizeof...(Integers)>=tensor_iterator_dim)>>
	BCINLINE
	void  operator ()(Integers... indicies) const {
		if (mask(indicies...))
			left(indicies...);
	}

	template<
		class... Integers,
		class=std::enable_if_t<(
				sizeof...(Integers)>=tensor_iterator_dim)>>
	BCINLINE
	void operator ()(Integers... indicies) {
		if (mask(indicies...))
			left(indicies...);
	}

public:

	BCINLINE bc::size_t size() const { return left.size(); }
	BCINLINE bc::size_t rows() const { return left.rows(); }
	BCINLINE bc::size_t cols() const { return left.cols(); }
	BCINLINE bc::size_t dim(int i) const { return left.dim(i); }
	BCINLINE auto inner_shape() const { return left.inner_shape(); }
};

}
}
}

#endif
