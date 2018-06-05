/*
 * binary_tree.h
 *
 *  Created on: Apr 18, 2018
 *      Author: joseph
 */

#ifndef BINARY_TREE_H_
#define BINARY_TREE_H_

#include "BC_Collections.h"

namespace BC {
namespace Structure {

template<class T, class deleter = default_deleter>
class binary_tree : Collection<T, deleter> {

	struct node {
		static constexpr deleter destroy = deleter();

		T internal;
		node* left;
		node* right;

		node(T d, node* l=nullptr, node* r=nullptr) :
			internal(d), left(l), right(r) {}
		~node() {
			deleter()(internal);
		}
	};

	int sz = 0;
	node* head = nullptr;


public:
	binary_tree() = default;
	virtual ~binary_tree() { clear(); }
	virtual bool empty() const { return !sz; }
	virtual int size()  const { return sz; }
	virtual bool add(T internal) override {

		if (!head) {
			head = new node(internal);
			++sz;
			return true;
		}
		node* ref = head;

		while (ref) {
			if (internal < ref->internal)
				if (ref -> left)
					ref = ref->left;
				else
					ref -> left = new node(internal);
			else if (internal > ref->internal)
				if (ref -> right)
					ref = ref->right;
				else
					ref -> right = new node(internal);
			else if (internal == ref->internal)
				return false;
		}
		++sz;
		return true;
	}
private:
	node* left_most_child(node* ref){
		while (ref->left && ref->right) {
			while (ref->left)
				ref = ref->left;
			if (ref->right)
				ref = ref->right;
		}

		return ref;
	}
	node* right_most_child(node* ref){
		while (ref->left && ref->right) {
			while (ref->right)
				ref = ref->right;
			if (ref->left)
				ref = ref->left;

		}

		return ref;
	}

	//naive implementation, always concats left most side
	node* remove_impl(T internal, node* ref) {
		if (!ref)
			return nullptr;
		if (ref->internal == internal){
			node* link;

			if (!ref->left)
				link =  ref->right;
			else if(!ref->right)
				link = ref->left;
			else
				link = left_most_child(ref);

			delete ref;
			return link;
		}
		if (internal < ref->internal)
			ref->left = remove_impl(internal, ref->left);
		else if (internal > ref->internal)
			ref->right = remove_impl(internal, ref->right);

		return ref;
	}

public:
	virtual bool remove(T internal) {
		remove_impl(internal, head);
		return true;
	}

private:
	void clear_impl(node* ref) {
		if (ref->left)
			clear_impl(ref->left);

		if (ref->right)
			clear_impl(ref->right);

		delete ref;
	}
public:
	virtual void clear() override {
		clear_impl(head);
		head = nullptr;
		sz = 0;
	}
private:
	void print_impl(node* ref) {
		if (!ref)
			return;

		print_impl(ref->left);
		std::cout << ref->internal << std::endl;
		print_impl(ref->right);
	}
public:
	void print() {
		print_impl(head);
	}
};







}

}



#endif /* BINARY_TREE_H_ */
