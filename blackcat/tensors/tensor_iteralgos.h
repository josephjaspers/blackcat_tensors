	self_type& fill(value_type value)
	{
		bc::algorithms::fill(
			this->get_stream(),
			this->cw_begin(),
			this->cw_end(),
			value);

		return *this;
	}

	self_type& zero() { return fill(0); }
	self_type& ones() { return fill(1); }

	template<class Function>
	void for_each(Function func) const {
		*this = this->un_expr(func);
	}

	template<class Function>
	void for_each(Function func) {
		*this = this->un_expr(func);
	}

	self_type& sort()
	{
		bc::algorithms::sort(
			this->get_stream(),
			this->cw_begin(),
			this->cw_end());

		return *this;
	}

	void randomize(value_type lb=0, value_type ub=1)
	{
		static_assert(
			expression_type::tensor_iterator_dim == 0 ||
			expression_type::tensor_iterator_dim == 1,
			"randomize not available to non-continuous tensors");

		using Random = bc::random::Random<system_tag>;
		Random::randomize(
			this->get_stream(),
			this->expression_template(), lb, ub);
	}
