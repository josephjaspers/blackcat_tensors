
#ifndef BC_BLAS
#define BC_BLAS
namespace BC {
namespace BLAS {


	template<class T>
	void cblas_dot_vv(T* c, const T* a, const T* b, int m) {
			for (int i = 0; i < m; ++i)
				c[0] += a[i] * b[i];
	}

//	cblas_sgemm(CblasColMajor, TRANS_A, TRANS_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	template<class T>
	void cblas_xcorr_vv(T* C, int krnl_length, const T* krnl,  int img_length, T* img) {
		const int position = img_length - krnl_length +  1;
		for (int i = 0; i < position; ++i) {
			cblas_dot_vv(&C[i], krnl, &img[i], krnl_length);
		}
	}

	template<class T>
	void cblas_xcorr_mm(bool ColMajor, bool trans_a, bool trans_b, T* C, int ldc, T alpha, int m, int n,  const T* krnl, int lda, T beta, int k, int l, T* img, int ldb)
	{
		int m_positions = n - l + 1;

		for (int m_ = 0; m_ < m_positions; ++m_)
			cblas_xcorr_vv(C[m_ * ldc], m, krnl[lda * m_], k, img[ldb * m_]);
	}
	template<class T>
	void cblas_xcorr_mm_padded(bool ColMajor, bool trans_a, bool trans_b, T* C, T alpha, int m, int n,  const T* krnl, int lda, T beta, int k, int l, T* img, int ldb) {

	}

}
}
#endif
