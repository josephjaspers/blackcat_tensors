/*
 * IO_Utility.h
 *
 *  Created on: Jun 28, 2018
 *      Author: joseph
 */

#ifndef IO_UTILITY_H_
#define IO_UTILITY_H_

namespace BC {
namespace IO{

template<class T> T BC_from_string(std::string str);
template<> int BC_from_string<int>(std::string str) { return stoi(str); }
template<> float BC_from_string<float>(std::string str) { return stof(str); }
template<> double BC_from_string<double>(std::string str) { return stod(str); }
template<> long BC_from_string<long>(std::string str) { return stol(str); }
template<> unsigned BC_from_string<unsigned>(std::string str) { return stoi(str); }
template<> char BC_from_string<char>(std::string str) { return (str[0]); }
template<> std::string BC_from_string<std::string>(std::string str) { return (str); }


}
}




#endif /* IO_UTILITY_H_ */
