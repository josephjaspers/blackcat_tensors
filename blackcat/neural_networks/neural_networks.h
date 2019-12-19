
/*
 * BCNN_Global_Unifier.h
 *
 *  Created on: Jun 28, 2018
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_NETWORKNETWORKS_H_
#define BLACKCATTENSORS_NEURALNETWORKS_NETWORKNETWORKS_H_

#include "common.h"

#include "layer_cache.h"
#include "layer_loader.h"
#include "layer_manager.h"
#include "layer_chain.h"

#include "optimzers/stochastic_gradient_descent.h"
#include "optimzers/adam.h"
#include "optimzers/momentum.h"

#include "layers/layer_traits.h"
#include "layer_cache.h"
#include "layers/feedforward.h"
#include "layers/unaryfunction.h"
#include "layers/nonlinear.h"
#include "layers/softmax.h"
#include "layers/lstm.h"
#include "layers/output_layer.h"
#include "layers/logging_output_layer.h"
#include "layers/convolution.h"
#include "layers/convolution_reference.h"
#include "layers/max_pooling.h"
#include "layers/flatten.h"
#include "layers/recurrent.h"
#include "network.h"

#endif
