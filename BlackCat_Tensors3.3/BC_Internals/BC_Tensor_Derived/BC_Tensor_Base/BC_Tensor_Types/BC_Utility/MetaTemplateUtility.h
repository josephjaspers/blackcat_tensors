/*
 * MetaTemplate_Adhoc_types.h
 *
 *  Created on: May 17, 2018
 *      Author: joseph
 */

#ifndef METATEMPLATE_ADHOC_TYPES_H_
#define METATEMPLATE_ADHOC_TYPES_H_

namespace BC {

struct DISABLED;

template<bool b, class T> using BC_enabled_if = std::conditional_t<b, T, DISABLED>;


}



#endif /* METATEMPLATE_ADHOC_TYPES_H_ */
