/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_BATCHED_REDUCE__H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_BATCHED_REDUCE__H_

#include "Expression_Template_Base.h"
#include "Tree_Evaluator.h"
#include "blas_tools/Blas_tools.h"


namespace BC {
namespace tensors {
namespace exprs {

template<class SystemTag>
struct BatchedReduce: BC::oper::Add_Assign {};

template<class ArrayType, class SystemTag>
struct Unary_Expression<BatchedReduce<SystemTag>, ArrayType>:
	Expression_Base< Unary_Expression<BatchedReduce<SystemTag>, ArrayType>>,
	BatchedReduce<SystemTag> {

    using value_type = typename ArrayType::value_type;
    using system_tag = SystemTag;
    using operation = BatchedReduce<SystemTag>;

    static constexpr int tensor_dimension  = 1;
    static_assert(ArrayType::tensor_dimension >= 1, "Broadcasted sum requires a tesnor of dimension 2 at minimum");
    static constexpr int tensor_iterator_dimension = ArrayType::tensor_iterator_dimension-1;

    ArrayType array;

    Unary_Expression(ArrayType array_, operation op=operation()):
    	operation(op),
    	array(array_) {}

    template<class... integers> BCINLINE
    auto operator ()(integers... index) const {
    	static_assert(tensor_dimension <= 3, "BatchedReduce Max Dimensions are 3");

    	if (ArrayType::tensor_dimension == 2 ) {
    		value_type total = 0;
    		for (int i = 0; i < array.rows(); ++i) {
        			operation::operator()(total, array(i, index...));
    		}
    		return total;
    	}
    	else /* if (ArrayType::tensor_dimension == 3 ) */ {
			value_type total = 0;
			for (int i = 0; i < array.rows(); ++i) {
				for (int j = 0; j < array.cols(); ++j) {
						operation::operator()(total, array(i, j, index...));
				}
			}
			return total;
		}
    }

    BCINLINE
    auto operator [](size_t index) const {
    	static_assert(tensor_dimension <= 3, "BatchedReduce Max Dimensions are 3");
    	if (ArrayType::tensor_dimension == 2 ) {
    		value_type total = 0;
    		for (int i = 0; i < array.rows(); ++i) {
        			operation::operator()(total, array(i, index));
    		}
    		return total;
    	}
    	else /* if (ArrayType::tensor_dimension == 3 ) */ {
			value_type total = 0;
			for (int i = 0; i < array.rows(); ++i) {
				for (int j = 0; j < array.cols(); ++j) {
						operation::operator()(total, array(i, j, index));
				}
			}
			return total;
		}
    }

	BCINLINE
	auto inner_shape() const {
		return BC::utility::make_lambda_array<tensor_dimension>(
				[&](size_t i) {
			return i == 0 ? array.dimension(tensor_dimension - 1) :  1;
		});
	}

    BCINLINE BC::size_t size() const { return array.dimension(ArrayType::tensor_dimension-1); }
    BCINLINE BC::size_t rows() const { return size(); }
    BCINLINE BC::size_t cols() const { return 1; }
    BCINLINE BC::size_t dimension(int i) const { return i == 0 ? size() : 1; }

};


} //ns BC
} //ns exprs
} //ns tensors



#endif /* FUNCTION_DOT_H_ */
