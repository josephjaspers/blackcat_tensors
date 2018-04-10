#pragma once

namespace BC {
    static void copy(unsigned* save, const unsigned* data, unsigned sz) {
        for (int i = 0; i < sz; ++i) {
            save[i] = data[i];
        }
    }

    static void mult_sum(unsigned* save, const unsigned* base, const unsigned* data, unsigned data_sz) {
    	*save = *base;
    	for (int i = 0; i < data_sz; ++i) {
    		*save *= data[i];
    	}
    }
};
