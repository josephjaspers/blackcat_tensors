
class CPU {

	template<typename T, typename lv, typename rv>
	struct Pointwise_Operation_Buffer {

		int sz;

		rv* next;
		lv* curr;

		//inherited
		Pointwise_Operation_Buffer(lv* left, rv* right, int sz) {
			curr = left;
			next = right;
			this->sz = sz;
		}

		virtual T operator [](unsigned index) = 0;
	};
	template<typename T, typename lv, typename rv>
	struct dot : Operation<T, lv, rv> {
		int m1_rows;
		int m2_cols;
		int m1_cols;

		T* temporary_lv;
		T* evaluated_temporary;

		//overloaded operators if lv/rv are buffers
		T operator [](unsigned index) override final {
			return evaluated_temporary[index];
		}

	};

	template<typename T, typename lv, typename rv>
	struct mul : Operation<T, lv, rv> {
		T operator [](unsigned index) override final {
			return curr[index] * next[index];
		}
	};
	template<typename T>
	struct div : Pointwise_Operation_Buffer<T> {
		T operator [](unsigned index) override final {
			return curr[index] / next[index];
		}
	};
	template<typename T>
	struct add : Pointwise_Operation_Buffer<T> {
		T operator [](unsigned index) override final {
			return curr[index] + next[index];
		}
	};
	template<typename T>
	struct sub : Pointwise_Operation_Buffer<T> {
		T operator [](unsigned index) override final {
			return curr[index] - next[index];
		}
	};

	//-----------------------for the scalar operations ------------- "next" is always the scalar

	template<typename T>
	struct mul_scal : Pointwise_Operation_Buffer<T> {
		T operator [](unsigned index) override final {
			return curr[index] * (*next);
		}
	};
	template<typename T>
	struct div_scal : Pointwise_Operation_Buffer<T> {
		T operator [](unsigned index) override final {
			return curr[index] / (*next);
		}
	};
	template<typename T>
	struct add_scal : Pointwise_Operation_Buffer<T> {
		T operator [](unsigned index) override final {
			return curr[index] + (*next);
		}
	};
	template<typename T>
	class sub_scal : Pointwise_Operation_Buffer<T> {
		T operator [](unsigned index) override final {
			return curr[index] - (*next);
		}
	};
};
