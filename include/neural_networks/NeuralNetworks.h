
/*
 * BCNN_Global_Unifier.h
 *
 *  Created on: Jun 28, 2018
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_NETWORKNETWORKS_H_
#define BLACKCATTENSORS_NEURALNETWORKS_NETWORKNETWORKS_H_

#include "common.h"

#include "Layer_Cache.h"
#include "Layer_Loader.h"
#include "Layer_Manager.h"
#include "Layer_Chain.h"

#include "optimzers/Stochastic_Gradient_Descent.h"
#include "optimzers/Adam.h"
#include "optimzers/Momentum.h"

#include "layers/Layer_Traits.h"
#include "Layer_Cache.h"
#include "layers/FeedForward.h"
#include "layers/UnaryFunction.h"
#include "layers/Nonlinear.h"
#include "layers/Softmax.h"
#include "layers/LSTM.h"
#include "layers/LSTM_Optimized.h"
#include "layers/Output_Layer.h"
#include "layers/Logging_Output_Layer.h"
#include "layers/Convolution.h"
#include "layers/Convolution_Reference.h"
#include "layers/Max_Pooling.h"
#include "layers/Flatten.h"
#include "layers/Recurrent.h"
#include "Network.h"

#endif
