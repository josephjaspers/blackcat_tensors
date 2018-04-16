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

template<class T>
class forward_list {

	struct node {

		node(T d, node* n = nullptr)
		: data(d), next(n) {}

		T data;
		node* next = nullptr;
	};


	node* head = nullptr;
	int sz = 0;
public:

	void push(T data) {
		if (head) {
			node* h = new node(data, head);
			head = h;
		} else {
			head = new node(data);
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
		T data = (head->data);
		rm_front();
		return (data);
	}

	T& first() {
		return head->data;
	}
	const T& first() const {
		return head->data;
	}
	T& second() {
		return head->next->data;
	}
	const T& second() const {
		return head->next->data;
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
