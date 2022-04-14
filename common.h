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

#ifndef ALN
#define ALN 64
#endif
#define min(a,b) a<b?a:b

#define u64 long long unsigned int
#define REAL double
const static int word2vec_word_embedding_size = 300;
const static REAL lamda = -1.;


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

	u64* row_ptr;
	u64* col_inds;
	double* vals;
} puma_csr_double_t;

/* A simple CSC struct.  */
typedef struct {
	u64 num_rows;
	u64 num_cols;
	u64 nnz;

	u64* col_ptr;
	u64* row_inds;
	REAL* vals;
} puma_csc_double_t;