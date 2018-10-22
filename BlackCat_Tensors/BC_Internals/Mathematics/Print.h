/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_PRINTFUNCTIONS_H_
#define BC_PRINTFUNCTIONS_H_

#include <string>
#include <iostream>
#include "PrintSparse.h"

namespace BC {
namespace IO {


template<typename T, class int_ranks, class os>
static void printHelper(const T& ary, const int_ranks ranks, const os outer, int dimension, std::string indent, int printSpace) {


	--dimension;

	if (dimension > 1) {
		std::cout << indent << "--- --- --- --- --- " << dimension << " --- --- --- --- ---" << std::endl;
		auto adj_indent = indent + '\t'; //add a tab to the index

		for (int i = 0; i < ranks[dimension]; ++i) {
			printHelper(&ary[i * outer[dimension - 1]], ranks, outer, dimension, adj_indent, printSpace);
		}

		auto gap = to_string(dimension);
		auto str = std::string(" ", gap.length());

	} else if (dimension == 1) {
		std::cout << indent << "--- --- --- --- --- " << dimension << " --- --- --- --- ---" << std::endl;

		for (int j = 0; j < ranks[dimension - 1]; ++j) {
			std::cout << indent + indent + "[ ";

			for (int i = 0; i < ranks[dimension]; ++i) {
				//convert to string --- seems to not be working with long/ints?
				auto str = to_string(ary[i * outer[dimension - 1] + j]);

				//if the string is longer than the printspace truncate it
				str = str.substr(0, (int)str.length() < printSpace ? str.length() : printSpace);

				//if its less we add blanks (so all numbers are uniform in size)
				if ((int)str.length() < printSpace)
					str += std::string(" ", printSpace - str.length());

				//print the string
				std::cout << str;

				//add some formatting fluff
				if (i < ranks[dimension] - 1)
					std::cout << " | ";

			}
			std::cout << " ]";
			std::cout << std::endl;
		}
	} else {
		std::cout << "[ ";
		for (int i = 0; i < ranks[dimension]; ++i) {
			//convert to string --- seems to not be working with long/ints?
			auto str = to_string(ary[i]);

			//if the string is longer than the printspace truncate it
			str = str.substr(0, (int)str.length() < printSpace ? str.length() : printSpace);

			//if its less we add blanks (so all numbers are uniform in size)
			if ((int)str.length() < printSpace)
				str += std::string(" ", printSpace - str.length());

			//print the string
			std::cout << str;

			//add some formatting fluff
			if (i < ranks[dimension] - 1)
				std::cout << " | ";
		}
		std::cout << " ]";
	}
}

template<typename T, class RANKS, class os>
static void print(const T& ary, const RANKS ranks, const os outer, int dimension, int print_length) {
	//if scalar
	if (dimension == 0) {
		std::cout << "[" << ary[0]  << "]" << std::endl;
		return;
	//if vector
	} else if (dimension == 1) {
		std::cout << "[ ";
		for (int i = 0; i < ranks[0]; ++i) {
			std::cout << ary[i * outer[0]];
			if (i != ranks[0] - 1) {
				std::cout << " | ";
			}
		}
		std::cout << " ]" << std::endl;
		return;
	}

	//else highier dimension
	std::string indent = "";
	printHelper(ary, ranks, outer, dimension, indent, print_length);
	std::cout << std::endl;
}
}
}

#endif /* BC_PRINTFUNCTIONS_H_ */
