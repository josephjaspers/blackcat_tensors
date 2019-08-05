/*
 * common.h
 *
 *  Created on: Jun 8, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORKS_COMMON_H_
#define BLACKCAT_NEURALNETWORKS_COMMON_H_

namespace BC {
namespace nn {

#ifndef BLACKCAT_DEFAULT_SYSTEM
#define BLACKCAT_DEFAULT_SYSTEM_T BC::host_tag
#else
#define BLACKCAT_DEFAULT_SYSTEM_T BC::BLACKCAT_DEFAULT_SYSTEM##_tag
#endif

}
}



#endif /* COMMON_H_ */
