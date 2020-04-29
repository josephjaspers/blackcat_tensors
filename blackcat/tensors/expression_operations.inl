#define BC_SCALAR_COEFFICIENTWISE_DEF(op, op_functor)                         \
    template<                                                                 \
        class ScalarType,                                                     \
        class=std::enable_if_t<                                               \
                std::is_convertible<ScalarType, value_type>::value>>          \
    auto op (const ScalarType& param) const                                   \
    {                                                                         \
        return bi_expr(oper::op_functor(),                                    \
                exprs::make_scalar_constant<system_tag>((value_type)param));  \
    }                                                                         \
                                                                              \
    template<                                                                 \
        class ScalarType,                                                     \
        class=std::enable_if_t<                                               \
                std::is_convertible<ScalarType, value_type>::value>>          \
    friend auto op (const ScalarType& param, const Expression_Base& tensor)   \
    {                                                                         \
        value_type value = param;                                             \
        auto scalar = exprs::make_scalar_constant<system_tag>(value);         \
        return make_expression(scalar).bi_expr(oper:: op_functor (), tensor); \
    }

#define BC_COEFFICIENTWISE_DEF(op, op_functor)             \
    template<class Xpr>                                    \
    auto op (const Expression_Base<Xpr>& param) const      \
    {                                                      \
        assert_valid(param);                               \
        return bi_expr(oper::op_functor(), param);         \
    }                                                      \
                                                           \
    BC_SCALAR_COEFFICIENTWISE_DEF(op, op_functor)          \


	BC_COEFFICIENTWISE_DEF(operator +, Add)
	BC_COEFFICIENTWISE_DEF(operator -, Sub)
	BC_COEFFICIENTWISE_DEF(operator %, Mul)
	BC_COEFFICIENTWISE_DEF(operator /, Div)
	BC_COEFFICIENTWISE_DEF(operator  == , Equal )
	BC_COEFFICIENTWISE_DEF(operator  >  , Greater)
	BC_COEFFICIENTWISE_DEF(operator  <  , Lesser)
	BC_COEFFICIENTWISE_DEF(operator  >= , Greater_Equal)
	BC_COEFFICIENTWISE_DEF(operator  <= , Lesser_Equal )
	BC_COEFFICIENTWISE_DEF(operator  && , And )
	BC_COEFFICIENTWISE_DEF(operator  || , Or )

	BC_COEFFICIENTWISE_DEF(approx_equal, Approx_Equal)
	BC_COEFFICIENTWISE_DEF(max_value, Max)
	BC_COEFFICIENTWISE_DEF(min_value, Min)
	BC_SCALAR_COEFFICIENTWISE_DEF(operator *, Scalar_Mul)

#undef BC_SCALAR_COEFFICIENTWISE_DEF
#undef BC_COEFFICIENTWISE_DEF

	template<class Xpr>
	auto operator *(const Expression_Base<Xpr>& param) const {

		using blas_traits = exprs::blas_expression_traits<expression_type>;
		using rv_blas_traits = exprs::blas_expression_traits<Xpr>;

		constexpr bool lv_trans = blas_traits::is_transposed::value;
		constexpr bool rv_trans = rv_blas_traits::is_transposed::value;

		constexpr bool scalmul = tensor_dim == 0 || Xpr::tensor_dim == 0;
		constexpr bool gemm    = tensor_dim == 2 && Xpr::tensor_dim == 2;
		constexpr bool gemv    = tensor_dim == 2 && Xpr::tensor_dim == 1;
		constexpr bool ger     = tensor_dim == 1 && Xpr::tensor_dim == 1 &&
				!lv_trans && rv_trans;
		constexpr bool dot     = tensor_dim == 1 && Xpr::tensor_dim == 1 &&
				!lv_trans && !rv_trans;

		using matmul_t =
				std::conditional_t<scalmul, oper::Scalar_Mul,
				std::conditional_t<gemm, oper::gemm<system_tag>,
				std::conditional_t<gemv, oper::gemv<system_tag>,
				std::conditional_t<ger,  oper::ger<system_tag>,
				std::conditional_t<dot,  oper::dot<system_tag>, void>>>>>;

		static_assert(!std::is_void<matmul_t>::value,
				"INVALID USE OF OPERATOR *");
		return bi_expr(matmul_t(), param);
	}

	// ---- Unary Expressions ---- //

	const auto transpose() const {
		return make_expression(make_transpose(this->expression_template()));
	}

	      auto transpose()       {
		return make_expression(make_transpose(this->expression_template()));
	}

	const auto t() const { return this->transpose(); }
	      auto t()       { return this->transpose(); }

	auto operator - () const {
		return un_expr(oper::negation);
	}

	// ---- expression_factory ---- //

	template<class functor>
	auto un_expr(functor f) const
	{
		return make_expression(
			exprs::make_un_expr(
				this->expression_template(), f));
	}

	template<
		class Functor,
		class Xpr,
		class=std::enable_if_t<
			exprs::expression_traits<Xpr>::is_expression_template::value>>
	auto bi_expr(Functor func, const Xpr& rv) const
	{
		return make_expression(
			exprs::make_bin_expr(
				this->expression_template(),
				rv.expression_template(),
				func));
	}

	template<class Xpr>
	void assert_valid(const Expression_Base<Xpr>& tensor) const
	{
		static_assert(std::is_same<system_tag, typename Xpr::system_tag>::value,
				"Tensor arguments must have compatible (same) system_tags");

		bool same_dim = tensor_dim == Xpr::tensor_dim;
		bool same_shape = this->inner_shape() == tensor.inner_shape();
		bool cwise_op = same_dim && same_shape;

		bool scalar_op = tensor_dim == 0 || Xpr::tensor_dim == 0;
		bool valid_broadcast_op = !same_dim && !cwise_op && valid_slice(tensor);
		bool valid_cwise_op = (same_dim && same_shape);

		if (!scalar_op && !valid_broadcast_op && !valid_cwise_op) {
			throw std::invalid_argument(
				"Tensor by Tensor operation error: shape mismatch."
				"\nthis->tensor_dim  = " + std::to_string(tensor_dim) +
				"\nthis->size()      = " + std::to_string(this->size()) +
				"\nthis_dims         = " + this->inner_shape().to_string() +
				"\nparam->tensor_dim = " + std::to_string(Xpr::tensor_dim) +
				"\nparam.size()      = " + std::to_string(tensor.size()) +
				"\nparam_dims        = " + tensor.inner_shape().to_string()
			);
		}
	}

	template<class Xpr>
	bool valid_slice(const Expression_Base<Xpr>& tensor) const {
		constexpr bc::size_t min_dim = traits::min(tensor_dim, Xpr::tensor_dim);

		for (int i = 0; i < min_dim; ++i)
			if (tensor.dim(i) != this->dim(i))
				return false;
		return true;
	}
