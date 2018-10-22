/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PRINTSPARSE_H_
#define PRINTSPARSE_H_

#include <string>

struct BC_to_string {
	template<class T>
	std::string operator () (const T& type)  const {
		return std::to_string(type);
	}
	std::string operator () (const std::string& type) const {
		return type;
	}
	std::string operator () (const char& type) const {
		return std::string(&type);
	}
};

static constexpr BC_to_string to_string = BC_to_string();



namespace BC {
namespace IO {
template<typename T, class int_ranks, class os>
static void printHelperSparse(const T& ary, const int_ranks ranks, const os outer, int dimension, std::string indent, int printSpace) {
	--dimension;

	if (dimension > 1) {
		std::cout << indent << "--- --- --- --- --- " << dimension << " --- --- --- --- ---" << std::endl;
		auto adj_indent = indent + '\t'; //add a tab to the index

		for (int i = 0; i < ranks[dimension]; ++i) {
			printHelperSparse(&ary[i * outer[dimension - 1]], ranks, outer, dimension, adj_indent, printSpace);
		}

		auto gap = to_string(dimension);
		auto str = std::string(" ", gap.length());
		std::cout << indent << "--- --- --- --- --- " << dimension << " --- --- --- --- ---" << std::endl;

	} else if (dimension == 1) {
		std::cout << indent << "--- --- --- --- --- " << dimension << " --- --- --- --- ---" << std::endl;

		for (int j = 0; j < ranks[dimension - 1]; ++j) {
			std::cout << indent + indent + "[ ";

			for (int i = 0; i < ranks[dimension]; ++i) {
				//convert to string --- seems to not be working with long/ints?
				auto str = to_string(ary[i * outer[dimension - 1] + j]);

				//if the string is longer than the printspace truncate it
				str = str.substr(0, str.length() < (unsigned)printSpace ? str.length() : printSpace);

				//if its less we add blanks (so all numbers are uniform in size)
				if (str.length() < (unsigned)printSpace)
					str += std::string(" ", printSpace - str.length());

				//print the string
				if (ary[i * outer[dimension - 1] + j] < .0001) {
					for (unsigned x= 0; x < str.length(); ++x) {
						std::cout << " ";
					}
				} else  {
					std::cout << str;
				}

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
			str = str.substr(0, str.length() < (unsigned)printSpace ? str.length() : printSpace);

			//if its less we add blanks (so all numbers are uniform in size)
			if (str.length() < (unsigned)printSpace)
				str += std::string(" ", printSpace - str.length());

			//print the string
			if (ary[i] < .0001) {
				for (unsigned x= 0; x < str.length(); ++x) {
						std::cout << " ";
					}
			} else
			std::cout << str;

			//add some formatting fluff
			if (i < ranks[dimension] - 1)
				std::cout << " | ";
		}
		std::cout << " ]";
	}
}

template<typename T, class RANKS, class os>
static void printSparse(const T& ary, const RANKS ranks, const os outer, int dimension, int print_length) {

	//If the tensor is a scalar
	if (dimension == 0) {
		std::cout << "[" << ary[0]  << "]" << std::endl;
		return;


	//If the tensor is a Vector
	} else if (dimension == 1) {
		std::cout << "[ ";
		for (int i = 0; i < ranks[0]; ++i) {
			if (ary[i] > .0001)
			std::cout << ary[i];
			else {
				std::cout << " ";
			}
			if (i != ranks[0] - 1) {
				std::cout << " | ";
			}
		}
		std::cout << " ]" << std::endl;
		return;
	}
	//Else the tensor is a matrix or higher-dimension
	std::string indent = "";
	printHelperSparse(ary, ranks, outer, dimension, indent, print_length);
	std::cout << std::endl;
}
}
}

#endif /* PRINTSPARSE_H_ */
