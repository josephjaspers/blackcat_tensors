
	std::string to_raw_string(int precision=8) const {
		return this->to_string(precision, false, false);
	}

	void print(int precision=8, bool pretty=true, bool sparse=false) const {
		std::cout << this->to_string(precision, pretty, sparse) << std::endl;
	}

	void print_sparse(int precision=8, bool pretty=true) const {
		std::cout << this->to_string(precision, pretty, true) << std::endl;
	}

	void raw_print(int precision=0, bool sparse=false) const {
		std::cout << this->to_string(precision, false, sparse) << std::endl;
	}

	void print_dims() const {
		for (int i = 0; i < tensor_dim; ++i) {
			std::cout << "[" << this->dim(i) << "]";
		}
		std::cout << std::endl;
	}

	void print_leading_dims() const {
		for (int i = 0; i < tensor_dim; ++i) {
			std::cout << "[" << this->leading_dim(i) << "]";
		}
		std::cout << std::endl;
	}