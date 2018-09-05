/*
 * PTEE_Binary_Linear.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef PTEE_BINARY_LINEAR_H_
#define PTEE_BINARY_LINEAR_H_

namespace BC {
namespace internal {
namespace tree {

template<class lv, class rv, class op>
struct evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_linear_op<op>()>> {
	static constexpr bool trivial_blas_evaluation = evaluator<lv>::trivial_blas_evaluation && evaluator<rv>::trivial_blas_evaluation;
	static constexpr bool trivial_blas_injection = evaluator<lv>::trivial_blas_injection || evaluator<rv>::trivial_blas_injection;
	static constexpr bool non_trivial_blas_injection = evaluator<lv>::non_trivial_blas_injection || evaluator<rv>::non_trivial_blas_injection;

	struct full_eval {
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			evaluator<lv>::linear_evaluation(branch.left, tensor);
			evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
			return tensor.data();
		}
	};
	struct left_eval {
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			evaluator<lv>::linear_evaluation(branch.left, tensor);
			return evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
		}
	};
	struct right_eval {
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			evaluator<rv>::linear_evaluation(branch.left, update_injection<op>(tensor));
			return evaluator<lv>::linear_evaluation(branch.left, tensor);
		}
	};

	template<class core, int a, int b>
	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
		static constexpr bool double_eval = evaluator<lv>::trivial_blas_evaluation && evaluator<rv>::trivial_blas_evaluation;

		using impl = std::conditional_t<double_eval, full_eval,
			std::conditional_t<evaluator<lv>::trivial_blas_evaluation, left_eval, right_eval>>;

		return impl::function(branch, tensor);
	}

	//------------------------------------------------------------------------
	struct left_trivial_injection {
		struct trivial_injection {
			template<class l, class r>
			static auto function(const l& left, const r& right) {
				return left;
			}
		};
		struct non_trivial_injection {
				template<class l, class r>
				static auto function(const l& left, const r& right) {
					return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
				}
			};
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			auto left = evaluator<lv>::injection(branch.left, tensor);
			auto right = evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));

			using impl = std::conditional_t<evaluator<rv>::trivial_blas_evaluation,
					trivial_injection, non_trivial_injection>;

			return impl::function(left, right);
		}
	};
	struct right_trivial_injection {
		struct trivial_injection {
			template<class l, class r>
			static auto function(const l& left, const r& right) {
				return right;
			}
		};
		struct non_trivial_injection {
				template<class l, class r>
				static auto function(const l& left, const r& right) {
					return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
				}
			};
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			auto left = evaluator<lv>::linear_evaluation(branch.left, tensor);
			auto right = evaluator<rv>::injection(branch.right, update_injection<op>(tensor));

			using impl = std::conditional_t<evaluator<lv>::trivial_blas_evaluation,
					trivial_injection, non_trivial_injection>;

			return impl::function(left, right);		}
	};

	struct left_nontrivial_injection {
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			auto left = evaluator<lv>::injection(branch.left, tensor);
			auto right = branch.right; //rv
			return binary_expression<std::decay_t<decltype(left)>, rv, op>(left, right);
		}
	};
	struct right_nontrivial_injection {
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			auto left = branch.left; //lv
			auto right = evaluator<rv>::injection(branch.right, tensor);
			return binary_expression<lv, std::decay_t<decltype(right)>, op>(left, right);
		}
	};
	template<class core, int a, int b>
	static auto injection(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
		static constexpr bool full_eval_b= evaluator<lv>::trivial_blas_evaluation && evaluator<rv>::trivial_blas_evaluation;

		using impl =
				std::conditional_t<full_eval_b, full_eval,
				std::conditional_t<evaluator<lv>::trivial_blas_injection, left_trivial_injection,
				std::conditional_t<evaluator<rv>::trivial_blas_injection, right_trivial_injection,
				std::conditional_t<evaluator<lv>::non_trivial_blas_injection, left_nontrivial_injection,
				std::conditional_t<evaluator<rv>::non_trivial_blas_injection, right_nontrivial_injection, void>>>>>;
		return impl::function(branch, tensor);
	}
	struct replacement_required {
		static auto function(const binary_expression<lv,rv,op>& branch) {
			using branch_t = binary_expression<lv,rv,op>;
			auto tmp =  temporary<internal::Array<branch_t::DIMS(), scalar_of<branch_t>, mathlib_of<branch_t>>>(branch.inner_shape());
			auto inject_tmp = injector<std::decay_t<decltype(tmp)>, 1, 0>(tmp);
			return injection(branch, inject_tmp);
		}
	};
	struct replacement_not_required {
		static auto function(const binary_expression<lv,rv,op>& branch) {
			return branch;
		}
	};
	static auto replacement(const binary_expression<lv,rv,op>& branch) {
		using impl = std::conditional_t<non_trivial_blas_injection, replacement_required, replacement_not_required>;
		return impl::function(branch);
	}


	static void destroy_temporaries(const binary_expression<lv, rv, op>& branch) {
		evaluator<lv>::destroy_temporaries(branch.left);
		evaluator<rv>::destroy_temporaries(branch.right);
	}

};


}
}
}



#endif /* PTEE_BINARY_LINEAR_H_ */
