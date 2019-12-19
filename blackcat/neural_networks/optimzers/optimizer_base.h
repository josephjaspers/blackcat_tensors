/*
 * Adam.h
 *
 *  Created on: Dec 11, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_OPTIMIZERS_OPTIMIZER_BASE_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_OPTIMIZERS_OPTIMIZER_BASE_H_

#include "optimizer_base.h"

namespace BC {
namespace nn {

struct Optimizer_Base {
	void save(Layer_Loader& loader, std::string name) {}
	void load(Layer_Loader& loader, std::string name) {}
};

}
}




#endif /* ADAM_H_ */
