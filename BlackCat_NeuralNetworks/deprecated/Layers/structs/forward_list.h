/*
 * frwd_list.h
 *
 *  Created on: Apr 10, 2018
 *      Author: joseph
 */

#ifndef FRWD_LIST_H_
#define FRWD_LIST_H_

namespace BC {
namespace NN {
namespace Structure {
//Taken from BC_Structures
template<class T>
class forward_list {

	struct node {

		node(T d, node* n = nullptr)
		: internal(d), next(n) {}

		T internal;
		node* next = nullptr;
	};


	node* head = nullptr;
	int sz = 0;
public:

	void push(T internal) {
		if (head) {
			node* h = new node(internal, head);
			head = h;
		} else {
			head = new node(internal);
		}
		sz++;
	}
	bool isEmpty() const {
		return sz == 0;
	}
	int size() const {
		 return sz;
	}
	void rm_front() {
		if (!head)
			return;

		node* h = head;
		head = head->next;
		--sz;

		delete h;
	}
	T pop() {
		T internal = (head->internal);
		rm_front();
		return (internal);
	}

	T& first() {
		return head->internal;
	}
	const T& first() const {
		return head->internal;
	}
	T& second() {
		return head->next->internal;
	}
	const T& second() const {
		return head->next->internal;
	}


	void clear() {
		node* ref = head;
		while(ref) {
			node* del = ref;
			ref = ref->next;
			delete del;
		}
		head = nullptr;
		sz = 0;
	}

	~forward_list() {
		clear();
	}
};

}
}
}



#endif /* FRWD_LIST_H_ */
