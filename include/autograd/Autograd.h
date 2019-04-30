/*
 * Autograd.h
 *
 *  Created on: Apr 22, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_AUTOGRAD_H_
#define BLACKCAT_AUTOGRAD_H_

namespace BC {
namespace autograd {

template<class SystemTag, int Dims, class ValueType>
class Weight {

	BC::Tensor<Dims, ValueType, Polymorphuc_Allocator<SystemTag, ValueType>>;


};



































sz            nb

template<int Dims, class ValueType>
class Variable {

};


}
}



#endif /* AUTOGRAD_H_ */
