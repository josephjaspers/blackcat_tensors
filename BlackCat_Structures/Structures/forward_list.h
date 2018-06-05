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
		: internal(d), next(n) {}

		T internal;
		node* next = nullptr;
	};


	node* head = nullptr;

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

	bool push(T internal) {
		if (head) {
			node* h = new node(internal, head);
			head = h;
		} else {
			head = new node(internal);
		}
		return true;
	}
	int contains(const T& internal) const  {
		node* ref = head;
		int index = 0;
		while(ref) {
			if (ref->internal == internal)
				return index;

			ref = ref->next;
			index++;
		}
		return 0;
	}
	T& get(int i) {
		node* ref = head;
		while (ref && i) {
			ref = ref->next;
			--i;
		}

		return ref->internal;
	}

	bool add(T internal) override {
		return push(internal);
	}
	bool empty() const override {
		return head == nullptr;
	}
	int size() const override {
		 node* ref = head;
		 int sz = 0;
		 while (ref) {
			 sz++;
			 ref = ref->next;
		 }
		 return sz;

	}
	//decapitate
	void remove_head() {
		if (!head)
			return;

		node* h = head;
		head = head->next;

		delete h;
	}
	T pop() {
		T internal = (head->internal);
		remove_head();
		return (internal);
	}

	T& first() {
		return head->internal;
	}
	//same as first
	T& front() {
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

	template<class functor>
	void for_each(functor f) const {
		node* ref = head;
		while(ref) {
			f(ref->internal);
			ref = ref->next;
		}
	}
	template<class functor>
	void for_each(functor f) {
		node* ref = head;
		while(ref) {
			f(ref->internal);
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
	}

	~forward_list() {
		clear();
	}
};

}
}



#endif /* FRWD_LIST_H_ */
