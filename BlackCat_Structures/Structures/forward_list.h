/*
 * frwd_list.h
 *
 *  Created on: Apr 10, 2018
 *      Author: joseph
 */

#ifndef FRWD_LIST_H_
#define FRWD_LIST_H_

namespace BC {
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
	node* tail = nullptr;
	int sz = 0;
public:

	void push_back(T data) {
		if (head) {
			tail->next = new node(data);
			tail = tail->next;
		} else {
			head = new node(data);
			tail = head;
		}
		sz++;
	}
	void push_front(T data) {
		if (head) {
			node* new_head = new node(data);
			new_head->next = head;
			head = new_head;
		} else {
			head = new node(data);
			tail = head;
		}
		sz++;
	}
	bool isEmpty() const {
		return sz == 0;
	}
	int size() const {
		 return size;
	}
	void rm_front() {
		node* h = head;
		head = head->next;
		--sz;

		if (sz == 0)
			tail = nullptr;

		delete h;
	}
	T pop_front() {
		T data = std::move(head->data);
		rm_front();
		return std::move(data);
	}

	T& front() {
		return head->data;
	}
	const T& front() const {
		return head->data;
	}

	void clear() {
		node* ref = head;
		while(ref) {
			node* del = ref;
			ref = ref->next;
			delete del;
		}
		head = nullptr;
		tail = nullptr;
		sz = 0;
	}

	~forward_list() {
		clear();
	}
};

}
}



#endif /* FRWD_LIST_H_ */
