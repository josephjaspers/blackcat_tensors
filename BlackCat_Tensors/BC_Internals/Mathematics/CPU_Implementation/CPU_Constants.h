/*
 * CPU_Constants.h
 *
 *  Created on: Jun 10, 2018
 *      Author: joseph
 */

#ifndef CPU_CONSTANTS_H_
#define CPU_CONSTANTS_H_

namespace BC {

template<class core_lib>
struct CPU_Constants {

    template<class U, class T, class V>
    static void scalar_mul(U& eval, T a, V b) {
        eval = get_value(a) * get_value(b);
    }

    template<class T>
    static T static_allocate(T value) {
        return T(value);
    }
};


}


#endif /* CPU_CONSTANTS_H_ */
