#include "BLACKCAT_CPU_MATHEMATICS.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

 //Pointwise increments ------------- COL Major --- for degree > 2 Ten

template<typename number_type>
    void CPU_MATHEMATICS<number_type>::copy(number_type* s, const unsigned* ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD) {

	--order;
	unsigned store_index = 0;
	unsigned m1_index = 0;

	if (order  == 0) {
		copy(s, m1, ranks[order]);

	} else {

		for (unsigned i = 0; i < ranks[order]; ++i) {
			copy(&s[store_index], ranks, order, s_LD, &m1[m1_index], m1_LD);
			store_index += s_LD[order];
			m1_index += m1_LD[order];
		}
	}
}
template<typename number_type>
    void CPU_MATHEMATICS<number_type>::fill(number_type* s, const unsigned* s_ranks,  unsigned order, const unsigned *s_LD, number_type m1) {
	if (order == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1;
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			fill(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, m1);
		}
	}
}


//----
template<typename number_type>
    void CPU_MATHEMATICS<number_type>::power(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD,
    																															 const number_type* m2, const unsigned* m2_LD) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = pow(m1[i], m2[i]);
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			power(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, &m2[m2_LD[order - 1] * i], m2_LD);
		}
	}
}

template<typename number_type>
    void CPU_MATHEMATICS<number_type>::multiply(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD,
    																															 const number_type* m2, const unsigned* m2_LD) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1[i] * m2[i];
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			multiply(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, &m2[m2_LD[order - 1] * i], m2_LD);
		}
	}
}

template<typename number_type>
    void CPU_MATHEMATICS<number_type>::divide(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD,
    																															 const number_type* m2, const unsigned* m2_LD) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1[i] / m2[i];
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			divide(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, &m2[m2_LD[order - 1] * i], m2_LD);
		}
	}
}

template<typename number_type>
    void CPU_MATHEMATICS<number_type>::add(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD,
    																															 const number_type* m2, const unsigned* m2_LD) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1[i] + m2[i];
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			add(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, &m2[m2_LD[order - 1] * i], m2_LD);
		}
	}
}


template<typename number_type>
    void CPU_MATHEMATICS<number_type>::subtract(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1, const unsigned* m1_LD,
    																															 const number_type* m2, const unsigned* m2_LD) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1[i] - m2[i];
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			subtract(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, &m2[m2_LD[order - 1] * i], m2_LD);
		}
	}
}

//-----------------------------------------------Scalar Methods ----------------------------------------------------//
template<typename number_type>
void CPU_MATHEMATICS<number_type>::power	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD,
																			const number_type* m1,  const unsigned* m1_LD, const number_type scal) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = pow(m1[i], scal);
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			power(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, scal);
		}
	}
}
template<typename number_type>
void CPU_MATHEMATICS<number_type>::multiply(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD,
																			const number_type* m1,  const unsigned* m1_LD, const number_type scal) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1[i] * scal;
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			multiply(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, scal);
		}
	}
}
template<typename number_type>
void CPU_MATHEMATICS<number_type>::divide	(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,
																			const unsigned* m1_LD, const number_type scal) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1[i] / scal;
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			divide(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, scal);
		}
	}
}
template<typename number_type>
void CPU_MATHEMATICS<number_type>::add		(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,
																			const unsigned* m1_LD, const number_type scal) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1[i] + scal;
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			add(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, scal);
		}
	}
}
template<typename number_type>
void CPU_MATHEMATICS<number_type>::subtract(number_type* s, const unsigned* s_ranks, unsigned order, const unsigned *s_LD, const number_type* m1,
																			const unsigned* m1_LD, const number_type scal) {
	if (order  == 1) {
		for (unsigned i = 0; i < s_ranks[0]; ++i) {
			s[i] = m1[i] - scal;
		}
	} else {
		for (unsigned i = 0; i < s_ranks[order - 1]; ++i) {
			subtract(&s[s_LD[order - 1] * i], s_ranks, order - 1, s_LD, &m1[m1_LD[order - 1] * i], m1_LD, scal);
		}
	}
}

