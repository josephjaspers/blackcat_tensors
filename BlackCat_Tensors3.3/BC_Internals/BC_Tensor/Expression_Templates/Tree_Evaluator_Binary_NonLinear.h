/*
 * PTEE_Binary_NonLinear.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef PTEE_BINARY_NONLINEAR_H_
#define PTEE_BINARY_NONLINEAR_H_

namespace BC {
namespace internal {
namespace tree {

template<class lv, class rv, class op>
struct evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_nonlinear_op<op>()>> {
	static constexpr bool trivial_blas_evaluation = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = evaluator<lv>::non_trivial_blas_injection || evaluator<rv>::non_trivial_blas_injection;


	template<class core, int a, int b>
	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
		return branch;
	}


	struct left_trivial_injection {
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			auto left = evaluator<lv>::injection(branch.left, tensor);
			auto right = branch.right;
			return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
		}
	};
	struct right_trivial_injection {
		template<class core, int a, int b>
		static auto function(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
			auto left = branch.left;
			auto right = evaluator<rv>::injection(branch.right, tensor);
			return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
		}
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
			//dont need to update injection
			//trivial injection left_hand side (we attempt to prefer trivial injections opposed to non-trivial)
			using impl  =
				std::conditional_t<evaluator<lv>::trivial_blas_injection, 		left_trivial_injection,
				std::conditional_t<evaluator<rv>::trivial_blas_injection, 		right_trivial_injection,
				std::conditional_t<evaluator<lv>::non_trivial_blas_injection, 	left_nontrivial_injection,
				std::conditional_t<evaluator<rv>::non_trivial_blas_injection, 	right_nontrivial_injection, void>>>>;

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




#endif /* PTEE_BINARY_NONLINEAR_H_ */
