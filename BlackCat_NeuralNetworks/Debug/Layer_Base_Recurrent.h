/*
 * Layer_Recurrent_Base.h
 *
 *  Created on: Aug 26, 2018
 *      Author: joseph
 */

#ifndef LAYER_RECURRENT_BASE_H_
#define LAYER_RECURRENT_BASE_H_

#include "../BC_Internals/Layers/Layer_Base.h"

namespace BC {
namespace NN {

template<class derived>
class Layer_Base_Recurrent : Layer_Base<derived> {

	int max_bptt = 36;
	int curr_time_stamp = 0;

	Layer_Base_Recurrent(int inputs) : Layer_Base<derived>(inputs) {
		static_assert(!std::is_same<void, std::declval<derived>().set_max_bptt_length(0)>::value,
				"RECURRENT LAYERS MUST IMPLEMENT void set_max_bptt_length(int ) const");
	}

	int max_numb_backprop_storages() const {
		return max_bptt;
	}

	int current_time_stamp() const {
		return curr_time_stamp;
	}

	void time_forward() const {
		curr_time_stamp += 1;
		curr_time_stamp %= max_bptt;
	}

	void time_backward() const {
		curr_time_stamp -= 1;

		if (curr_time_stamp < 0)
			curr_time_stamp = max_bptt;

	}

};

}
}




#endif /* LAYER_RECURRENT_BASE_H_ */
