/*
 * string_extensions.h
 *
 *  Created on: Nov 11, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_STRING_EXTENSIONS_H_
#define BLACKCAT_TENSORS_STRING_EXTENSIONS_H_

#include <string>
#include <algorithm>

namespace BC {

/**
 * Inherits from std::string.
 * Simply defines additional "extensions" to std::string.
 */
struct string:
		std::string {

	using std::string::string;

	int count(char value) const {
		return std::count(this->begin(), this->end(), value);
	}

	bool startswith(const std::string& str) const {
		return this->size() >= str.size() &&
				this->substr(0, str.size()) == str;
	}

	bool endswith(const BC::string& str) const {
		if (this->size() < str.size())
			return false;

		auto start_idx = this->size() - str.size();
		return this->substr(start_idx, str.size()) == str;
	}

	std::vector<BC::string> split(char delim) const {
		std::vector<BC::string> splts;

		auto curr = this->begin();
		auto end = this->end();

		while (curr < end) {
			auto next = std::find(curr, end, delim);

			if (curr < end)
				splts.push_back(BC::string(curr, next));

			curr = next + 1;
		}

		return splts;
	}
};


}



#endif /* STRING_EXTENSIONS_H_ */
