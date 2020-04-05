using parent_type::assert_valid;

private:


	template<class ScalarType>
	using enable_if_scalar = std::enable_if_t<
			std::is_convertible<ScalarType, value_type>::value>;


public:


	template<class Xpr> BCHOT
	self_type& operator = (const Tensor_Base<Xpr>& param)
	{
		static_assert(tensor_dim >= Xpr::tensor_dim,
				"BlackCat_Tensors: Operator= is not a valid operation for (reduction) broadcasting");
		BC_ASSERT_ASSIGNABLE(
				"self_type& operator = (const Tensor_Base<Xpr>& param)");
		assert_valid(param);
		return evaluate(assignment_bi_expr(bc::oper::assign, param));
	}

#define BC_OPER_BASIC_ASSIGNMENT_DEF(op, op_functor)                       \
                                                                           \
    template<class Xpr> BCHOT                                              \
    self_type& operator op (const Tensor_Base<Xpr>& param) {               \
        BC_ASSERT_ASSIGNABLE(                                              \
                "operator " #op "(const Tensor_Base<Xpr>& param)");        \
        assert_valid(param);                                               \
        using operation = std::conditional_t<                              \
                (tensor_dim >= Xpr::tensor_dim),                           \
                oper::op_functor##_Assign,                                 \
                oper::Atomic_##op_functor<system_tag>>;                    \
        return evaluate(assignment_bi_expr(operation(), param));           \
    }                                                                      \

#define BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)                      \
                                                                           \
    template<class ScalarType, class=enable_if_scalar<ScalarType>>         \
    self_type& operator  op (const ScalarType& param) {                    \
        BC_ASSERT_ASSIGNABLE(                                              \
                "operator " #op " (const Tensor_Base<Xpr>& param)");       \
        value_type value = param;                                          \
        return evaluate(assignment_bi_expr(                                \
                        oper:: op_functor##_Assign(),                      \
                        exprs::make_scalar_constant<system_tag>(value)));  \
    }                                                                      \

#define BC_OPER_ASSIGNMENT_DEF(op, op_functor)           \
        BC_OPER_SCALAR_ASSIGNMENT_DEF(op, op_functor)    \
        BC_OPER_BASIC_ASSIGNMENT_DEF(op, op_functor)


	template<class ScalarType, class=enable_if_scalar<ScalarType>>
	self_type& operator = (const ScalarType& param)
	{
		BC_ASSERT_ASSIGNABLE(
				"self_type& operator =(const Tensor_Base<Xpr>& param)");
		return evaluate(assignment_bi_expr(
				bc::oper::assign,
				exprs::make_scalar_constant<system_tag>((value_type)param)));
	}

	BC_OPER_ASSIGNMENT_DEF(+=, Add)
	BC_OPER_ASSIGNMENT_DEF(-=, Sub)
	BC_OPER_ASSIGNMENT_DEF(%=, Mul)
	BC_OPER_ASSIGNMENT_DEF(/=, Div)


	struct Alias; friend struct Alias;
	struct Alias {

		self_type& tensor;

		template<class Xpr>
		auto& operator = (const Tensor_Base<Xpr>& param) {
			tensor.assert_valid(param);
			return tensor.evaluate(
					tensor.assignment_bi_expr(oper::alias_assign, param));
		}

		template<class Xpr>
		auto& operator += (const Tensor_Base<Xpr>& param) {
			tensor.assert_valid(param);
			return tensor.evaluate(
					tensor.assignment_bi_expr(oper::alias_add_assign, param));
		}

		template<class Xpr>
		auto& operator -= (const Tensor_Base<Xpr>& param) {
			tensor.assert_valid(param);
			return tensor.evaluate(
					tensor.assignment_bi_expr(oper::alias_sub_assign, param));
		}
	};

	Alias alias() {
		return Alias { *this };
	}

private:

	template<
		class Functor,
		class Xpr,
		class=std::enable_if_t<
				exprs::expression_traits<Xpr>::is_expression_template::value>>
	auto assignment_bi_expr(Functor func, const Xpr& rv)
	{
		return make_tensor(
				exprs::make_bin_expr(this->expression_template(), rv.expression_template(), func));
	}

	template<class Xpr>
	self_type& evaluate(const Tensor_Base<Xpr>& tensor) {
		BC_ASSERT(this->get_stream().get_allocator().allocated_bytes() == 0,
				"Evaluation expects streams allocate_bytes to be 0 pre-evaluation");

		exprs::evaluate(tensor.expression_template(), this->get_stream());

		BC_ASSERT(this->get_stream().get_allocator().allocated_bytes() == 0,
				"Evaluation expects streams allocate_bytes to be 0 post-evaluation");
		return *this;
	}
public:

#undef BC_OPER_ASSIGNMENT_DEF
#undef BC_OPER_SCALAR_ASSIGNMENT_DEF
#undef BC_OPER_BASIC_ASSIGNMENT_DEF

