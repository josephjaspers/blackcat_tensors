#include "LinearAlgebraRoutines.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"


template<typename number_type>
void Tensor_Operations<number_type>::correlation_autoPadding(number_type * s, const number_type* filter, unsigned filt_length, const number_type* img, unsigned img_rows, unsigned img_cols) {
    //Convert everything to signed int [need arithmatic with negative values] 
    const int FILT_LENGTH = filt_length;
    const int IMG_ROWS = img_rows;
    const int IMG_COLS = img_cols;
    const int padding = (FILT_LENGTH - 1) / 2;

    for (int r = -padding; r < IMG_ROWS - padding; ++r) {
        for (int c = -padding; c < IMG_COLS - padding; ++c) {
            number_type tmp = 0;
            for (int filt_r = 0; filt_r < FILT_LENGTH; ++filt_r) {
                for (int filt_c = 0; filt_c < FILT_LENGTH; ++filt_c) {
                    int img_row = r + filt_r;
                    int img_col = c + filt_c;
                    if (img_row >= 0 && img_col >= 0 && img_row < IMG_ROWS && img_col < IMG_COLS) {
                        tmp += filter[filt_r * FILT_LENGTH + filt_c] * img[img_row * IMG_COLS + img_col];
                    }
                }
            }
            s[(r + padding) * IMG_COLS + c + padding] = tmp;
        }
    }
}

template<typename number_type>
void Tensor_Operations<number_type>::convolution_autoPadding(number_type * s, const number_type* filter, unsigned filt_length, const number_type* img, unsigned img_rows, unsigned img_cols) {
    //Convert everything to signed int [need arithmatic with negative values] 
    const int FILT_LENGTH = filt_length;
    const int IMG_ROWS = img_rows;
    const int IMG_COLS = img_cols;
    const int padding = (FILT_LENGTH - 1) / 2;


    for (int r = -padding; r < IMG_ROWS - padding; ++r) {
        for (int c = -padding; c < IMG_COLS - padding; ++c) {
            number_type tmp = 0;
            for (int filt_r = 0; filt_r < FILT_LENGTH; ++filt_r) {
                for (int filt_c = 0; filt_c < FILT_LENGTH; ++filt_c) {
                    int img_index_row = r + filt_r;
                    int img_index_col = c + filt_c;
                    int filt_index_row = FILT_LENGTH - filt_r;
                    int filt_index_col = FILT_LENGTH - filt_c;

                    if (img_index_row >= 0 && img_index_col >= 0 && img_index_row < IMG_ROWS && img_index_col < IMG_COLS) {

                        int FILT_INDEX = filt_index_row * FILT_LENGTH + filt_index_col;
                        int IMG_INDEX = img_index_row * IMG_COLS + img_index_col;

                        tmp += filter[FILT_INDEX] * img[IMG_INDEX];
                    }
                }
            }
            s[(r + padding) * IMG_COLS + c + padding] = tmp;
        }
    }
}

template<typename number_type>
void Tensor_Operations<number_type>::correlation(number_type* s, const number_type* filter, unsigned filter_dim, const number_type* img, unsigned img_dim) {
    unsigned save_dim = img_dim - filter_dim + 1;

    for (unsigned r = 0; r < save_dim; ++r) {
        for (unsigned c = 0; c < save_dim; ++c) {

            unsigned save_index = r * save_dim + c;
            unsigned img_base_index = r * img_dim + c;
            s[save_index] = 0;

            for (unsigned filt_r = 0; filt_r < filter_dim; ++filt_r) {

                unsigned img_ROW = filt_r * img_dim;
                unsigned filter_ROW = filt_r * filter_dim;

                for (unsigned filt_c = 0; filt_c < filter_dim; ++filt_c) {
                    s[save_index] += filter[filter_ROW + filt_c] * img[img_base_index + img_ROW + filt_c];
                }
            }
        }
    }
}

template<typename number_type>
void Tensor_Operations<number_type>::convolution(number_type* s, const number_type* filter, unsigned filter_dim, const number_type* img, unsigned img_dim) {

    unsigned save_dim = img_dim - filter_dim + 1;

    for (unsigned r = 0; r < save_dim; ++r) {
        for (unsigned c = 0; c < save_dim; ++c) {

            unsigned save_index = r * save_dim + c;
            unsigned img_base_index = r * img_dim + c;
            s[save_index] = 0;

            for (unsigned filt_r = 0; filt_r < filter_dim; ++filt_r) {

                unsigned img_ROW = filt_r * img_dim;
                unsigned filter_ROW = filter_dim - filt_r;


                for (unsigned filt_c = 0; filt_c < filter_dim; ++filt_c) {
                    unsigned filter_COL = filter_dim - filt_c - 1;
                    s[save_index] += filter[filter_ROW + filter_COL] * img[img_base_index + img_ROW + filt_c];
                }
            }
        }
    }
}
