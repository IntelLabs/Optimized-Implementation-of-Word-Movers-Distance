#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <bits/stdc++.h>
#include<iomanip>

#include <sys/types.h>
#include <sys/timeb.h>

#if 0
#include "ittnotify.h"

#endif

#define u64 long long unsigned int

using namespace std;

#ifndef ALN
#define ALN 64
#endif
#define min(a,b) a<b?a:b

#define REAL double

double *r_sel;
double *v_sel;

u64 v_r = 0;
/******************************* Global Variables ************************/
const static int word2vec_word_embedding_size = 300;
const static REAL lamda = -1.;
int max_iter = 1;

/******************Common Data Structure *****************/

typedef struct connector_t {
	u64 u, v;
	double edge_weight;

} connector_t;

/* A simple CSR struct.  */
typedef struct {
	u64 num_rows;
	u64 num_cols;
	u64 nnz;

	u64 *row_ptr;
	u64 *col_inds;
	double *vals;
} puma_csr_double_t;

/* A simple CSC struct.  */
typedef struct {
	u64 num_rows;
	u64 num_cols;
	u64 nnz;

	u64 *col_ptr;
	u64 *row_inds;
	REAL *vals;
} puma_csc_double_t;

/******************************* Global Variables ************************/

puma_csr_double_t c_csr;
//puma_csc_double_t w_csc;

double *r_arr;
double *vecs;
u64 nmtp_threads = 1;

/******************************* Timing Function **************************/
unsigned long long getMilliCount() {
	timeb tb;
	ftime(&tb);
	unsigned long long nCount = tb.millitm + (tb.time & 0xfffffffff) * 1000;
	return nCount;
}
unsigned long long getMilliSpan(unsigned long long nTimeStart) {
	long long int nSpan = getMilliCount() - nTimeStart;
	if (nSpan < 0)
		nSpan += 0x100000 * 1000;
	return nSpan;
}

#define BLOCK_SIZE 16
void cdist_gemm_dense(REAL *restrict c, double const *const u,
    double const *const v, const u64 M, const u64 N, const u64 K) {
  for (u64 j = 0; j < N; j += BLOCK_SIZE)
  {
    const u64 j_end = min(j+BLOCK_SIZE, N);
    for (u64 i = 0; i < M; i += BLOCK_SIZE)
    {
      const u64 i_end = min(i+BLOCK_SIZE, M);
      for (u64 k = 0; k < K; k += BLOCK_SIZE)
      {
        const u64 k_end = min(k+BLOCK_SIZE, K);
        for (u64 jj = j; jj < j_end; jj++) {
          for (u64 ii = i; ii < i_end; ii++) {
            REAL s = 0.;
            for (u64 kk = k; kk < k_end; kk++) {
              const REAL d = u[ii * K + kk] - v[jj * K + kk];
              s += (d * d);
            }
            c[jj * M + ii] += s;
          }
        }
      }
    }
  }
}
/*************************** Utilitty Functions **************************/
REAL inline euclidean_distance(double const *const u, double const *const v) {
	REAL s = 0.0;
#pragma unroll
	for (u64 i = 0; i < word2vec_word_embedding_size; ++i) {
		const REAL d = u[i] - v[i];
		s += (d * d);
	}
	return sqrt(s);
}

void inline select_non_empty_entries(u64 data_vocab_size, double const * const restrict r,
		double * restrict r_sel, double * restrict v_sel,
		double const * const restrict vecs, u64 &v_r) {
	for (u64 i = 0; i < data_vocab_size; i++)
	{
		if (r[i] == 0)
		continue;
		r_sel[v_r] = 1.0/r[i];

		// copy those words
		double *v_sel_ptr = v_sel + v_r * word2vec_word_embedding_size;
		const double *vecs_ptr = vecs + i * word2vec_word_embedding_size;
#pragma omp parallel for
		for (u64 k = 0; k < word2vec_word_embedding_size; k++) {
			v_sel_ptr[k] = vecs_ptr[k];
		}
		v_r++;
	}
}

void print_matrix(double *X, u64 rows, u64 cols) {
	for (u64 i = 0; i < rows; i++) {

		for (u64 j = 0; j < cols; j++) {

			cout << X[i * cols + j] << "\n";

		}
	}
}

void print_csr(u64 num_rows, double *vals, u64 *col_inds, u64 *row_ptr) {
	for (u64 i = 0; i < num_rows; i++) {
		double sum = 0.;
		for (u64 e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
			u64 j = col_inds[e];
			//[Vxnum_docs]
			cout << i << "," << j << "," << vals[e] << endl;

		}
	}
}

static inline u64 p_binary_search(
u64 const *const restrict weights,
u64 left,
u64 right,
u64 const query) {
	while ((right - left) > 0) {
		u64 mid = left + ((right - left) >> 1);

		if (weights[mid] <= query && weights[mid + 1] > query) {
			return mid;
		}

		if (weights[mid] < query) {
			left = mid + 1;
		} else {
			right = mid;
		}
	}

	return right;
}

/*************************** Utilitty Functions **************************/
void inline sinkhorn_wmd(const double *r, const double *vecs, REAL *WMD,
		const REAL lamda, const int max_iter, const u64 data_vocab_size,
		const u64 num_docs, const u64 word2vec_word_embedding_size) {

	// select non-zero word embedding
	/************************************************************************/

	select_non_empty_entries(data_vocab_size, r, r_sel, v_sel, vecs, v_r);

	//cout<<nmtp_threads<<endl;

	/************************************************************************/
	// M = cdist(vecs[sel], vecs).astype(np.float64)
    cout << "volcabulary size in the input doc: " << v_r << endl;
#ifdef VERBOSE
	cout << "volcabulary size in the input doc: " << v_r << endl;
	cout << "num docs: " << num_docs << endl;
#endif
	REAL *M;
	posix_memalign((void**) &M, ALN, (v_r * data_vocab_size) * sizeof(REAL));
	assert(M != NULL);

	//# K=exp(-lambda * M)
	//  K = np.exp(- M * lamb)
	REAL *K;
	posix_memalign((void**) &K, ALN, (v_r * data_vocab_size) * sizeof(REAL));
	assert(K != NULL);

	
	REAL *x;
	posix_memalign((void**) &x, ALN, (v_r * num_docs) * sizeof(REAL));
	assert(x != NULL);

	// u=1/x
	REAL *u;
	posix_memalign((void**) &u, ALN, (v_r * num_docs) * sizeof(REAL));
	assert(u != NULL);

	REAL *K_over_r;
	posix_memalign((void**) &K_over_r, ALN,
			(data_vocab_size * v_r) * sizeof(REAL));
	assert(K_over_r != NULL);

    #pragma omp parallel
    {
        #pragma omp barrier
    }
    unsigned long long start = 0, end = 0;
	start = getMilliCount();
    u64 const *csr_row_ptr = c_csr.row_ptr;
	u64 const *csr_cols = c_csr.col_inds;
	double const *csr_vals = c_csr.vals;
    u64 total_items_csr = c_csr.nnz;
    u64 csr_item_per_mtp = (total_items_csr + nmtp_threads - 1)
        / nmtp_threads;
    u64* mtps_start_item_csr, *mtps_end_item_csr, *row_id_start;
    mtps_start_item_csr=new u64[nmtp_threads];
    mtps_end_item_csr=new u64[nmtp_threads];
    row_id_start= new u64[nmtp_threads];
	// compute initial x
#pragma omp parallel for
	for (u64 t = 0; t < nmtp_threads; t++) {

        mtps_start_item_csr[t] = t * csr_item_per_mtp;
        mtps_end_item_csr[t] = min(mtps_start_item_csr[t] + csr_item_per_mtp,
        total_items_csr);
        row_id_start[t] = p_binary_search(csr_row_ptr, 0, data_vocab_size,
        mtps_start_item_csr[t]);
		u64 total_items_vrXDoc = num_docs * v_r;
		u64 item_per_mtp = (total_items_vrXDoc + nmtp_threads - 1)
				/ nmtp_threads;
		u64 mtps_start_item_vrXDoc = t * item_per_mtp;

		u64 mtps_end_item_vrXDoc = (min(mtps_start_item_vrXDoc + item_per_mtp,
				total_items_vrXDoc));

		for (u64 j = mtps_start_item_vrXDoc; j < mtps_end_item_vrXDoc; j++) {
			x[j] = 0.;
			u[j] = v_r;
		}

	}

	// compute distance between word embeding
	// compute K
#pragma omp parallel for
	for (u64 t = 0; t < nmtp_threads; t++) {
		u64 item_per_mtp = (data_vocab_size + nmtp_threads - 1) / nmtp_threads;
		u64 mtps_start_item = t * item_per_mtp;
		u64 end_point = min(mtps_start_item + item_per_mtp, data_vocab_size);
		for (u64 j = mtps_start_item; j < end_point; j++) {
			const u64 i_offset = j * v_r;

			REAL *M_ptr = M + i_offset;
			REAL *K_ptr = K + i_offset;
			REAL *K_over_r_ptr = K_over_r + i_offset;

			double const *const v_prt = vecs + j * word2vec_word_embedding_size;
			for (u64 i = 0; i < v_r; i++) {

				const REAL multiplier = r_sel[i];

				REAL const M_j = euclidean_distance(
						v_sel + i * word2vec_word_embedding_size, v_prt);
                M_ptr[i] = M_j;
				REAL const K_val = exp(M_j * lamda);
				K_ptr[i] = K_val;
				K_over_r_ptr[i] = multiplier * K_val;
				//K_transpose[j * v_r + i] = K_val;
			}
		}
	}
	/*
	 for (u64 i = 0; i < v_r; i++) {
	 for (u64 j = 0; j < 64; j++){
	 printf("%f\n", K[i * data_vocab_size+j]);
	 }
	 }
	 */

	/******************************************************************/

#if 0
	print_matrix(K_transpose, data_vocab_size, v_r);
	print_matrix(K_over_r, v_r, data_vocab_size);
	print_matrix(u, v_r, num_docs);
#endif

	/***************************Main Loop *********************************/
	// # while x changes: x=diag(1./r)*K*(c_csr.*(1./(K’*(1./x))))
	//unsigned long long start = 0, end = 0;
	

	int iteration = 0;
	while (iteration < max_iter) {
		//start = getMilliCount();

#pragma omp parallel for
		for (u64 t = 0; t < nmtp_threads; t++) {

			u64 row_id = row_id_start[t];
			u64 next_row_idx = csr_row_ptr[row_id + 1];

			 for (u64 idx = mtps_start_item_csr[t]; idx < mtps_end_item_csr[t];
                  idx++) {
				u64 const j = csr_cols[idx];
				REAL const val = csr_vals[idx];

				if (idx == next_row_idx) {

					while (idx >= csr_row_ptr[row_id + 1]) {
						++row_id;
					}
					next_row_idx = csr_row_ptr[row_id + 1];
				}
				REAL sum = 0.;

				REAL *KT_ptr = K + row_id * v_r;
				REAL *u_ptr = u + j * v_r;
				for (u64 k = 0; k < v_r; k++) {
					sum += (KT_ptr[k] * u_ptr[k]);
				}

				//[Vxnum_docs]
				REAL const output = ((val / sum));

				REAL *x_ptr = x + j * v_r;
				REAL *K_r = (K_over_r + row_id * v_r);
				for (u64 i = 0; i < v_r; i++) {
					const REAL multiplier = K_r[i];
					// using x for this purpose!

#pragma omp atomic

					//saving it while transposing on the fly!
					x_ptr[i] += multiplier * output;

				}

			}
		}
#if 0
		for (u64 i = 0; i < v_r; i++) {

			for (u64 j = 0; j < num_docs; j++) {

				cout << x[i * num_docs + j] << "\n";

			}
		}
#endif

#pragma omp parallel for
		for (u64 t = 0; t < nmtp_threads; t++) {
			u64 total_items_vrXDoc = num_docs * v_r;
			u64 item_per_mtp = (total_items_vrXDoc + nmtp_threads - 1)
					/ nmtp_threads;
			u64 mtps_start_item_vrXDoc = t * item_per_mtp;
			u64 mtps_end_item_vrXDoc = (min(
					mtps_start_item_vrXDoc + item_per_mtp, total_items_vrXDoc));

			for (u64 j = mtps_start_item_vrXDoc; j < mtps_end_item_vrXDoc;
					j++) {
				u[j] = 1.0 / x[j];
				x[j] = 0.;
			}

		}

#if 0
		for (u64 i = 0; i < v_r; i++) {

			for (u64 j = 0; j < num_docs; j++) {

				cout << u[i * num_docs + j] << "\n";

			}
		}
#endif
		iteration++;
		//end += getMilliSpan(start);
		//cout << "It=" << iteration << " |" << (double) end / 1000.0 << " s"
		//		<< endl;
	}

	//  u = 1.0 / x
	//  w_csr.T = (c_csr.multiply(1 / (K.T @ u)))T
#pragma omp parallel for
	for (u64 t = 0; t < nmtp_threads; t++) {
		u64 row_id = row_id_start[t];
		u64 next_row_idx = csr_row_ptr[row_id + 1];

		for (u64 idx = mtps_start_item_csr[t]; idx < mtps_end_item_csr[t];
                  idx++) {
			u64 const j = csr_cols[idx];
			REAL const val = csr_vals[idx];

			if (idx == next_row_idx) {

				while (idx >= csr_row_ptr[row_id + 1]) {
					++row_id;
				}
				next_row_idx = csr_row_ptr[row_id + 1];
			}
			double sum = 0.;
			REAL *KT_prt = (K + row_id * v_r);
			REAL *u_ptr = (u + j * v_r);
#pragma unroll
			for (u64 k = 0; k < v_r; k++) {
				sum += (KT_prt[k] * u_ptr[k]);
			}

			//[Vxnum_docs]
			const REAL output = ((val / sum));
			//cout<<output<<endl;
			REAL *x_ptr = x + j * v_r;
			for (u64 i = 0; i < v_r; i++) {
				const u64 index = i + row_id * v_r;
				const REAL multiplier = K[index] * M[index];
				// using x for this purpose!

#pragma omp atomic
				x_ptr[i] += multiplier * output;

			}

		}
		//#pragma omp barrier
	}

#pragma omp parallel for
	for (u64 j = 0; j < num_docs; j++) {
		REAL wmd_j = 0.0;
		REAL *u_ptr = u + j * v_r;
		REAL *x_ptr = x + j * v_r;
		for (u64 i = 0; i < v_r; i++) {
			//x[i * num_docs + j] = u[i * num_docs + j] * x[i * num_docs + j];
			REAL x_ij = u_ptr[i] * x_ptr[i];
			wmd_j += x_ij;
		}
		WMD[j] = wmd_j;
	}

    #pragma omp parallel
    {
        #pragma omp barrier
    }
    end = getMilliSpan(start);
	cout << "elapsed=" << (double) end / 1000.0 << endl;
	// free all allocated memory
	free(x);
	free(u);
	free(K);
	free(M);
	free(K_over_r);
	free(v_sel);
	free(r_sel);
}

void sinkhorn_wmd_cost(const int max_iter, const u64 data_vocab_size,
		const u64 num_docs, const u64 v_r, const u64 nmtp_threads, 
        const u64 word2vec_word_embedding_size) {

    
	
	// select non-zero word embedding
	/************************************************************************/

	//select_non_empty_entries(data_vocab_size, r, r_sel, v_sel, vecs, v_r);

    u64 read_bytes_step1 = 1 + (data_vocab_size + v_r * word2vec_word_embedding_size)*sizeof(double);
    u64 write_bytes_step1 = (v_r + v_r * word2vec_word_embedding_size)*sizeof(double) + 1;

	/************************************************************************/
	
	// compute distance between word embeding
	// compute K
    u64 read_bytes_step2 = (v_r * nmtp_threads
                          + v_r * word2vec_word_embedding_size 
                          + data_vocab_size * word2vec_word_embedding_size
                          + data_vocab_size * word2vec_word_embedding_size*v_r)*sizeof(double);
    u64 write_bytes_step2 = (v_r * data_vocab_size 
                           + v_r * data_vocab_size
                           + v_r * data_vocab_size)*sizeof(REAL);
                           
    u64 flops_step2 =   (v_r * data_vocab_size * (word2vec_word_embedding_size*3+21))
                       +(v_r * data_vocab_size * 21)
                       +(v_r * data_vocab_size * 1);                       
	/******************************************************************/
	// compute initial x
    u64 write_bytes_step3 = (v_r * num_docs * 2)*sizeof(REAL);
                                 
    u64 binsearch_read_cost = nmtp_threads * log2(c_csr.num_rows) * sizeof(c_csr.row_ptr[0]);
    u64 binsearch_write_cost = nmtp_threads *  sizeof(u64); 
	/***************************Main Loop *********************************/   
    u64 read_bytes_per_iteration = 0;
    u64 write_bytes_per_iteration = 0;
    u64 flop_per_iteration = 0;
    // we need to add all the i terms and multiply by #iterations    
    u64 read_bytes_step5i = (c_csr.num_rows * sizeof(c_csr.row_ptr[0])
                          + c_csr.nnz * sizeof(c_csr.vals[0]) 
                          + c_csr.nnz * sizeof(c_csr.col_inds[0])
                          )
                          +((c_csr.nnz * v_r  //data_vocab_size 
                          + v_r * c_csr.nnz   //num_docs
                          + c_csr.nnz * v_r ) //data_vocab_size
                          *sizeof(REAL));
                          
    u64 write_bytes_step5i = (v_r * c_csr.nnz)*sizeof(REAL);
 
    u64 flops_step5i = (v_r * c_csr.nnz * 4);                        

    u64 read_bytes_step6i = (num_docs * v_r *sizeof(REAL));
                          
    u64 write_bytes_step6i = (num_docs * v_r * 2 * sizeof(REAL));
 
    u64 flops_step6i = (v_r * num_docs * 20);    // 20 flops for div
    
    read_bytes_per_iteration = read_bytes_step5i + read_bytes_step6i;
    write_bytes_per_iteration = write_bytes_step5i + write_bytes_step6i;
    flop_per_iteration = flops_step5i + flops_step6i;
	// # while x changes: x=diag(1./r)*K*(c_csr.*(1./(K’*(1./x))))
	

	//  u = 1.0 / x
	//  w_csr.T = (c_csr.multiply(1 / (K.T @ u)))T
    u64 read_bytes_step7 = (c_csr.num_rows * sizeof(c_csr.row_ptr[0])
                          + c_csr.nnz * sizeof(c_csr.vals[0]) 
                          + c_csr.nnz * sizeof(c_csr.col_inds[0]))
                          +(c_csr.nnz * v_r  //data_vocab_size 
                          + v_r * c_csr.nnz   //num_docs
                          + c_csr.nnz * v_r * 2) //data_vocab_size
                          * sizeof(REAL);
                          
    u64 write_bytes_step7 = (v_r * c_csr.nnz)*sizeof(REAL);
 
    u64 flops_step7 = (v_r * c_csr.nnz * 5);    
    
    u64 read_bytes_step8 =(v_r * num_docs * 2 * sizeof(REAL));
                          
    u64 write_bytes_step8 = (num_docs) * sizeof(REAL);
 
    u64 flops_step8 = (v_r * num_docs * 2);
   
   
   u64 total_read_bytes = read_bytes_step1 
                        + read_bytes_step2
                        +  binsearch_read_cost 
                        + read_bytes_per_iteration*max_iter
                        + read_bytes_step7
                        + read_bytes_step8;
                        
   u64 total_write_bytes = write_bytes_step1 
                        + write_bytes_step2 
                        + write_bytes_step3 
                        + binsearch_write_cost
                        + write_bytes_per_iteration*max_iter
                        + write_bytes_step7
                        + write_bytes_step8;
                        
   u64 total_flops = flops_step2+flop_per_iteration*max_iter+flops_step7+flops_step8;
   
   u64 total_memory_footprint = (2*data_vocab_size 
   +2 * data_vocab_size * word2vec_word_embedding_size +  c_csr.nnz ) * sizeof(double)
   + (data_vocab_size * v_r * 3
   + v_r * num_docs * 2) * sizeof(REAL)
   + c_csr.nnz * sizeof(u64)
   + (data_vocab_size + 1) * sizeof(u64);
   
   
   //printf("memory=%llu B read=%llu write=%llu flops=%llu\n", 
   //total_memory_footprint, total_read_bytes, total_write_bytes, total_flops);
    double read_bw = 6.4 * nmtp_threads * 1000000000;
    double write_bw = read_bw * 0.5 ;
    double peak_flops = 2 * 64 * nmtp_threads * 1000000000;
    double time_bw = total_read_bytes/read_bw + total_write_bytes/write_bw;
    double time_flops = total_flops/peak_flops;
    printf("%llu %llu %llu %llu %llu %0.2f %0.2f %0.4f %0.9f %0.9f\n", 
    nmtp_threads, 
    total_memory_footprint, 
    total_read_bytes,
    total_write_bytes, 
    total_flops, 
    read_bw, 
    write_bw, 
    peak_flops, 
    time_bw, 
    time_flops);
    
}

void sinkhorn_wmd_cost_cached(const int max_iter, const u64 data_vocab_size,
		const u64 num_docs, const u64 v_r, const u64 nmtp_threads, 
        const u64 word2vec_word_embedding_size) {

	
	// select non-zero word embedding
	/************************************************************************/
  
	//select_non_empty_entries(data_vocab_size, r, r_sel, v_sel, vecs, v_r);

    u64 read_bytes_step1 = 1 + (data_vocab_size + v_r * word2vec_word_embedding_size)*sizeof(double);
    u64 write_bytes_step1 = (v_r + v_r * word2vec_word_embedding_size)*sizeof(double) + 1;

	/************************************************************************/
	
	// compute distance between word embeding
	// compute K
    u64 read_bytes_step2 = (v_r * nmtp_threads
                          + v_r * word2vec_word_embedding_size 
                          + data_vocab_size * word2vec_word_embedding_size
                          + data_vocab_size * word2vec_word_embedding_size*v_r/8)*sizeof(double);
    u64 write_bytes_step2 = (v_r * data_vocab_size 
                           + v_r * data_vocab_size
                           + v_r * data_vocab_size)*sizeof(REAL);
                           
    u64 flops_step2 =   (v_r * data_vocab_size * (word2vec_word_embedding_size*3+21))
                       +(v_r * data_vocab_size * 21)
                       +(v_r * data_vocab_size * 1);                       
	/******************************************************************/
	// compute initial x
    u64 write_bytes_step3 = (v_r * num_docs * 2)*sizeof(REAL);
                                 
    u64 binsearch_read_cost = nmtp_threads * log2(c_csr.num_rows) * sizeof(c_csr.row_ptr[0]);
    u64 binsearch_write_cost = nmtp_threads *  sizeof(u64); 
	/***************************Main Loop *********************************/   
    u64 read_bytes_per_iteration = 0;
    u64 write_bytes_per_iteration = 0;
    u64 flop_per_iteration = 0;
    // we need to add all the i terms and multiply by #iterations    
    u64 read_bytes_step5i = (c_csr.num_rows * sizeof(c_csr.row_ptr[0])
                          + c_csr.nnz * sizeof(c_csr.vals[0]) 
                          + c_csr.nnz * sizeof(c_csr.col_inds[0])
                          )
                          +((c_csr.nnz * v_r/16  //data_vocab_size 
                          + v_r * c_csr.nnz/16   //num_docs
                          + c_csr.nnz * v_r ) //data_vocab_size
                          *sizeof(REAL));
                          
    u64 write_bytes_step5i = (v_r * c_csr.nnz)*sizeof(REAL);
 
    u64 flops_step5i = (v_r * c_csr.nnz * 4);                        

    u64 read_bytes_step6i = (num_docs * v_r *sizeof(REAL));
                          
    u64 write_bytes_step6i = (num_docs * v_r * 2 * sizeof(REAL));
 
    u64 flops_step6i = (v_r * num_docs * 20);    // 20 flops for div
    
    read_bytes_per_iteration = read_bytes_step5i + read_bytes_step6i;
    write_bytes_per_iteration = write_bytes_step5i + write_bytes_step6i;
    flop_per_iteration = flops_step5i + flops_step6i;
	// # while x changes: x=diag(1./r)*K*(c_csr.*(1./(K’*(1./x))))
	

	//  u = 1.0 / x
	//  w_csr.T = (c_csr.multiply(1 / (K.T @ u)))T
    u64 read_bytes_step7 = (c_csr.num_rows * sizeof(c_csr.row_ptr[0])
                          + c_csr.nnz * sizeof(c_csr.vals[0]) 
                          + c_csr.nnz * sizeof(c_csr.col_inds[0]))
                          +(c_csr.nnz * v_r  //data_vocab_size 
                          + v_r * c_csr.nnz/16   //num_docs
                          + c_csr.nnz * v_r * 2/16) //data_vocab_size
                          * sizeof(REAL);
                          
    u64 write_bytes_step7 = (v_r * c_csr.nnz)*sizeof(REAL);
 
    u64 flops_step7 = (v_r * c_csr.nnz * 5);    
    
    u64 read_bytes_step8 =(v_r * num_docs * 2/16 * sizeof(REAL));
                          
    u64 write_bytes_step8 = (num_docs) * sizeof(REAL);
 
    u64 flops_step8 = (v_r * num_docs * 2);
   
   
   u64 total_read_bytes = read_bytes_step1 
                        + read_bytes_step2
                        +  binsearch_read_cost 
                        + read_bytes_per_iteration*max_iter
                        + read_bytes_step7
                        + read_bytes_step8;
                        
   u64 total_write_bytes = write_bytes_step1 
                        + write_bytes_step2 
                        + write_bytes_step3 
                        + binsearch_write_cost
                        + write_bytes_per_iteration*max_iter
                        + write_bytes_step7
                        + write_bytes_step8;
                        
   u64 total_flops = flops_step2+flop_per_iteration*max_iter+flops_step7+flops_step8;
   
   u64 total_memory_footprint = (2*data_vocab_size 
   +2 * data_vocab_size * word2vec_word_embedding_size +  c_csr.nnz ) * sizeof(double)
   + (data_vocab_size * v_r * 3
   + v_r * num_docs * 2) * sizeof(REAL)
   + c_csr.nnz * sizeof(u64)
   + (data_vocab_size + 1) * sizeof(u64);
   
   
   //printf("memory=%llu B read=%llu write=%llu flops=%llu\n", 
   //total_memory_footprint, total_read_bytes, total_write_bytes, total_flops);
    double read_bw = 6.4 * nmtp_threads * 1000000000;
    double write_bw = read_bw * 0.5 ;
    double peak_flops = 2 * 64 * nmtp_threads * 1000000000;
    double time_bw = total_read_bytes/read_bw + total_write_bytes/write_bw;
    double time_flops = total_flops/peak_flops;
    printf("%llu %llu %llu %llu %llu %0.2f %0.2f %0.4f %0.9f %0.9f\n", 
    nmtp_threads, 
    total_memory_footprint, 
    total_read_bytes,
    total_write_bytes, 
    total_flops, 
    read_bw, 
    write_bw, 
    peak_flops, 
    time_bw, 
    time_flops);
    
}
void main(int argc, char *argv[]) {
	const char *mat_filename = "./mat.mtx";

	cout << "Reading c_csr matrix csr: " << mat_filename << endl;
	ifstream mat_file( mat_filename);

	if (!mat_file.is_open()) {
		cout << "Could not open the file:" << mat_filename << " exiting\n";
		exit(0);
	}

	string line;
	mat_file >> c_csr.num_rows;
	mat_file >> c_csr.num_cols;
	mat_file >> c_csr.nnz;

#ifdef VERBOSE
	cout << "Running: " << argv[0] << ", V:" << c_csr.num_rows
			<< ", docs:" << c_csr.num_cols << ", occurances:" << c_csr.nnz << endl;
#endif
	// allocate memory for the graph
	posix_memalign((void**) &c_csr.row_ptr, ALN,
			(c_csr.num_rows + 1) * sizeof(u64));
	assert(c_csr.row_ptr != NULL);

	posix_memalign((void**) &c_csr.col_inds, ALN, (c_csr.nnz) * sizeof(u64));
	assert(c_csr.col_inds != NULL);

	posix_memalign((void**) &c_csr.vals, ALN, (c_csr.nnz) * sizeof(double));
	assert(c_csr.vals != NULL);

	u64 *degree;
	posix_memalign((void**) &degree, ALN, (c_csr.num_rows + 1) * sizeof(u64));
	assert(degree != NULL);

	connector_t *edgeList;
	posix_memalign((void**) &edgeList, ALN, (c_csr.nnz) * sizeof(connector_t));
	assert(edgeList != NULL);

#pragma omp parallel for
	for (u64 i = 0; i <= c_csr.num_rows; i++) {
		degree[i] = 0;
	}
	// take edges from file and initialize the graph and also compute the count
	// ree of each vertex
	u64 i = 0;
	u64 word1, word2;
	double w;
	while (!mat_file.eof() && i < c_csr.nnz) {
		mat_file >> word1;
		mat_file >> word2;
		mat_file >> w;

		edgeList[i].u = word1 - 1;
		edgeList[i].v = word2 - 1;
		edgeList[i].edge_weight = w;

		degree[word1 - 1]++;
		i++;
	}

	mat_file.close();
	cout << "nnz: " << i << " out of: " << c_csr.num_cols * c_csr.num_rows
			<< endl;

	c_csr.row_ptr[0] = 0;
	// compute prefix sum
	for (u64 i = 1; i <= c_csr.num_rows; i++) {
		c_csr.row_ptr[i] = c_csr.row_ptr[i - 1] + degree[i - 1];
		degree[i - 1] = 0;
	}
	// now copy the edges
	for (u64 i = 0; i < c_csr.nnz; i++) {
		u64 word1 = edgeList[i].u;
		u64 k = c_csr.row_ptr[word1] + degree[word1];
		c_csr.col_inds[k] = edgeList[i].v;
		c_csr.vals[k] = (double) edgeList[i].edge_weight;
		degree[word1]++;
	}

	free(edgeList);

	//////////////////////////////////////////////////////////////////
	const char *v_filename = "./vecs.out";
	cout << "reading vecs (sparse vector): " << v_filename << endl;
	ifstream v_file( v_filename);

	posix_memalign((void**) &vecs, ALN,
			(c_csr.num_rows * word2vec_word_embedding_size) * sizeof(double));
	assert(vecs != NULL);

	if (!v_file.is_open()) {
		cout << "Could not open the file:" << v_filename << " exiting\n";
		exit(0);
	}

	i = 0;
	for (u64 j = 0; j < c_csr.num_rows; j++) {
		getline(v_file, line);
		stringstream linestream( line);
		string data;
		for (u64 l = 0; l < word2vec_word_embedding_size; l++) {
			std
			::getline(linestream, data, ','); // read up-to the
			vecs[j * word2vec_word_embedding_size + l] = stod(data);
		}

	}
	v_file.close();

	// Now read from the r.out file: this is the word frequency in the input file.
	const char *r_filename = "./r.out";
	cout << "reading r (sparse vector): " << r_filename << endl;
	ifstream r_file( r_filename);

	posix_memalign((void**) &r_arr, ALN, (c_csr.num_rows) * sizeof(double));
	assert(r_arr != NULL);

	if (!r_file.is_open()) {
		cout << "Could not open the file:" << r_filename << " exiting\n";
		exit(0);
	}

	i = 0;
	double val;

	while (!r_file.eof()) {

		r_file >> val;
		r_arr[i] = val; // std::stof(data);
		//cout<<r_arr[i]<<endl;
		i++;
	}
	//cout << "r_arr length:" << i - 1 << endl;
	r_file.close();

	// select non-zero entry
	posix_memalign((void**) &r_sel, ALN, (c_csr.num_rows) * sizeof(double));
	assert(r_sel != NULL);

	posix_memalign((void**) &v_sel, ALN,
			(c_csr.num_rows * word2vec_word_embedding_size) * sizeof(double));
	assert(v_sel != NULL);

	REAL *WMD;
	posix_memalign((void**) &WMD, ALN, (c_csr.num_cols) * sizeof(REAL));
	assert(WMD != NULL);

#pragma omp parallel
	{
		nmtp_threads = omp_get_num_threads();
	}
	cout << "num_threads "<<nmtp_threads << endl;
    
#if 0
   __itt_domain* pD = __itt_domain_create( "My Domain" );
   __itt_frame_begin_v3(pD, NULL);
#endif
	// Main function.
	unsigned long long start = 0, end = 0;
	start = getMilliCount();
	//auto start_time = chrono::high_resolution_clock::now();
	sinkhorn_wmd(r_arr, vecs, WMD, lamda, max_iter, c_csr.num_rows,
			c_csr.num_cols, word2vec_word_embedding_size);

	end = getMilliSpan(start);
   #if 0
    __itt_frame_end_v3(pD, NULL);
    #endif
	//auto end_time = chrono::high_resolution_clock::now();
	//cout<<"elapsed=";
	//cout << chrono::duration_cast<chrono::seconds>(end_time - start_time).count() << ":";
	//cout << chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << "\n";
	cout << "elapsed=" << (double) end / 1000.0 << endl;
	// write output files in scores file
	ofstream outfile;
	outfile.open("scores", std::ofstream::out);
	for (u64 j = 0; j < c_csr.num_cols; j++) {
		outfile << std
	::setprecision(9) << WMD[j] << endl;
	//cout<<std::setprecision(9) << WMD[j] << endl;
}
outfile.close();
#if 0
 //int cores[]={1,2,4,8,16,32,64,128,256,4096,16384,32768,65536};
 /*
data_vocab_size	v_r	num_docs	nnz	            max_iter
100,000	        19	5000	    173087	            1
1,000,000	    19	5000	    173087	            1
10,000,000	    19	5000	    173087	            1
100,000,000	    19	5000	    173087	            1
100,000	        44	5000	    173087	            1
100,000	        44	5000	    173087	            15
100,000	        44	5000	    1730870	            15
100,000	        44	10000	    1730870	            15
200,000,000	    100	20,000,000	17,308,700,000	    15
4,000,000,000	100	20,000,000	17,308,700,000	    15
20,000,000,000	100	20,000,000	17,308,700,000	    15
40,000,000,000	100	20,000,000	17,308,700,000	    15
 
 
 */
 //int cores[]={1536,2048,2560};
 int cores[]={1,2,4,8,16,32,64,128,256,4096,16384,32768,65536};
 int num_cases = sizeof(cores)/sizeof(cores[0]);
 for(int i = 0; i<num_cases;i++ ){
/*     





Dataset7
Dataset8
Dataset9
Dataset10
Dataset11
Dataset12
*/
 /*    
 //Dataset1
 v_r=19;
 max_iter=1; 
 c_csr.num_rows = 100000;
 c_csr.num_cols=5000;
 c_csr.nnz = 173087;
 */
 
 /*
 //Dataset2
 v_r=19;
 max_iter=1; 
 c_csr.num_rows = 1000000;
 c_csr.num_cols=5000;
 c_csr.nnz = 173087;
 */
 
 /*
 //Dataset3
 v_r=19;
 max_iter=1; 
 c_csr.num_rows = 10000000;
 c_csr.num_cols=5000;
 c_csr.nnz = 173087;
 */
 
 /*
 //Dataset4
 v_r=19;
 max_iter=1; 
 c_csr.num_rows = 100000000;
 c_csr.num_cols=5000;
 c_csr.nnz = 173087;
*/

/*
 //Dataset5
 v_r=44;
 max_iter=1; 
 c_csr.num_rows = 100000;
 c_csr.num_cols=5000;
 c_csr.nnz = 173087;
 
 */
 /*
 //Dataset6
 v_r=44;
 max_iter=15; 
 c_csr.num_rows = 100000;
 c_csr.num_cols=5000;
 c_csr.nnz = 173087;
 */
 /*
 //Dataset7 100,000	44	5000	1730870	15

 v_r=44;
 max_iter=15; 
 c_csr.num_rows = 100000;
 c_csr.num_cols=5000;
 c_csr.nnz = 1730870;
 */
 
 
 //Dataset8 100,000	44	10000	1730870	15
 v_r=44;
 max_iter=15; 
 c_csr.num_rows = 100000;
 c_csr.num_cols=10000;
 c_csr.nnz = 1730870;
 
 /*
 v_r=100;
 max_iter=15; 
 c_csr.num_rows = 200000000;
 c_csr.num_cols =  20000000;
 c_csr.nnz =    17308700000;
 
 */
 
 /*
 v_r=100;
 max_iter=15; 
 c_csr.num_rows = 4000000000;
 c_csr.num_cols =   20000000;
 c_csr.nnz =     17308700000;
 */
 
 /*
 v_r=100;
 max_iter=15; 
 c_csr.num_rows = 20000000000;
 c_csr.num_cols =    20000000;
 c_csr.nnz =      17308700000;
 */
 /*
 v_r=100;
 max_iter=15; 
 c_csr.num_rows = 40000000000;
 c_csr.num_cols =    20000000;
 c_csr.nnz =      17308700000;
 */
 
 cout <<c_csr.num_rows<<" "<<v_r<< " " << c_csr.num_cols<<" "<<c_csr.nnz<<" "<<max_iter<<" ";

 sinkhorn_wmd_cost(max_iter, c_csr.num_rows,
 c_csr.num_cols, v_r, cores[i],
 word2vec_word_embedding_size);
 }
 u64 csr_data =  ( c_csr.nnz ) * sizeof(double) + c_csr.nnz * sizeof(u64) + (c_csr.num_rows + 1) * sizeof(u64);

 printf("%llu\n", csr_data);
 //print_csr(c_csr.num_rows, c_csr.vals, c_csr.col_inds, c_csr.row_ptr);
#endif
// free allocated memory
free(WMD);
free(vecs);
free(r_arr);

free(c_csr.col_inds);
free(c_csr.row_ptr);
free(c_csr.vals);

//free(w_csc.row_inds);
//free(w_csc.col_ptr);
//free(w_csc.vals);
}
