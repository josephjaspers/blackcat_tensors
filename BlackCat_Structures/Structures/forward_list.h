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

template<class T, class deleter = default_deleter>
class forward_list : Collection<T, deleter> {

	struct node {

		node(T d, node* n = nullptr)
		: data(d), next(n) {}

		T data;
		node* next = nullptr;
	};


	node* head = nullptr;
	int sz = 0;

public:

	forward_list() = default;
	forward_list(const forward_list& cpy) {
		node* ref = cpy.head;

		while (ref) {
			this->add(cpy.head);
			ref = ref->next;
		}
	}
	forward_list(forward_list&& cpy) {
		this->head = cpy.head;
		this->sz = cpy.sz;
		cpy.head = nullptr;
		cpy.sz = 0;
	}
	forward_list& operator = (const forward_list& cpy) {
		this->clear();
		node* ref = cpy.head;

		while (ref) {
			this->add(cpy.head);
			ref = ref->next;
		}
		return *this;
	}
	forward_list& operator = (forward_list&& cpy) {
		this->clear();
		this->head = cpy.head;
		this->sz = cpy.sz;
		cpy.head = nullptr;
		cpy.sz = 0;
		return *this;
	}

	bool push(T data) {
		if (head) {
			node* h = new node(data, head);
			head = h;
		} else {
			head = new node(data);
		}
		sz++;
		return true;
	}
	bool contains(const T& data) const  {
		node* ref = head;
		while(ref) {
			if (ref->data == data)
				return true;
			ref = ref->next;
		}
	}
	T* get(const T& data) {
		node* ref = head;
		while (ref) {
			if (ref->data == data)
				return &(ref->data);
			ref = ref->next;
		}
		return nullptr;
	}

	bool add(T data) override {
		return push(data);
	}
	bool empty() const override {
		return sz == 0;
	}
	int size() const override {
		 return sz;
	}
	//decapitate
	void remove_head() {
		if (!head)
			return;

		node* h = head;
		head = head->next;
		--sz;

		delete h;
	}
	T pop() {
		T data = (head->data);
		remove_head();
		return (data);
	}

	T& first() {
		return head->data;
	}
	//same as first
	T& front() {
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

	template<class functor>
	void for_each(functor f) const {
		node* ref = head;
		while(ref) {
			f(ref->data);
			ref = ref->next;
		}
	}
	template<class functor>
	void for_each(functor f) {
		node* ref = head;
		while(ref) {
			f(ref->data);
			ref = ref->next;
		}
	}

	void clear() override {
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



#endif /* FRWD_LIST_H_ */
