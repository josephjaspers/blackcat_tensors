/*
 * Utility.h
 *
 *  Created on: Sep 24, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_LAYERS_UTILITY_H_
#define BC_INTERNALS_LAYERS_UTILITY_H_

namespace BC {
namespace NN {
    struct circular_int  {

        int x = 0;
        int max = 0;

        circular_int() = default;
        circular_int(int max_) : x(0), max(max_) {}
        circular_int(int x_, int max_) : x(x_), max(max_) {}

        operator int () { return x; }

        int operator ++() {
            x++;

            if (x >= max)
                x = 0;

            return x;
        }
        int operator --() {
            x--;

            if (x < 0)
                x = max - 1;

            return x;
        }
        int operator ++(int) {
            int tmp_x = x;
            this->operator++();
            return tmp_x;
        }
        int operator --(int) {
            int tmp_x = x;
            this->operator--();
            return tmp_x;
        }

        int operator + (int value) {
            if (value + x < max)
                return value + x;
            else
                return (value + x) % max;
        }
        int operator - (int value) {
            if (x - value > 0)
                return x - value;
            else
                return std::abs(x - value) % max;
        }

    };

}
}



#endif /* BC_INTERNALS_LAYERS_UTILITY_H_ */
