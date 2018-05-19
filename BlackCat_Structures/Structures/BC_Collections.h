/*
 * BC_Collections.h
 *
 *  Created on: Apr 18, 2018
 *      Author: joseph
 */

#ifndef BC_COLLECTIONS_H_
#define BC_COLLECTIONS_H_

namespace BC {
namespace Structure {

struct default_deleter {
	template<class T>
	void operator()(T& t) const {
		return;
	}
};

template<class T, class deleter>
class Collection {
public:
	static constexpr deleter destroy = deleter();

	virtual bool empty() const= 0;
	virtual int  size()  const= 0;
	virtual void clear()= 0;
	virtual bool  add(T) = 0;

	virtual ~Collection() {};

};


template<class T> std::string to_str(const T& t) {
	 return std::to_string(t);
}
std::string to_str(std::string str) {
	return str;
}


}
}




#endif /* BC_COLLECTIONS_H_ */
