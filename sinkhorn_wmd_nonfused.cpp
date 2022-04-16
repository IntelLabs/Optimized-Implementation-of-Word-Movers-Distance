
#include "common.h"
using namespace std;



/******************************* Global Variables ************************/

int max_iter = 15;



/******************************* Global Variables ************************/

puma_csr_double_t c_csr;
//puma_csc_double_t w_csc;

double* r_arr;
double* vecs;

double* r_sel;
double* v_sel;
u64 v_r = 0;

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

/*************************** Utilitty Functions **************************/
double euclidean_distance(const double* u, const double* v) {
	double s = 0.0;
	u64 i;
	//#pragma unroll
	//#pragma vector
	for (i = 0; i < word2vec_word_embedding_size; ++i) {
		const double d = u[i] - v[i];
		s += (d * d);
	}
	return sqrt(s);
}

void select_non_empty_entries(u64 data_vocab_size, double const* const __restrict__ r,
	double* __restrict__ r_sel, double* __restrict__ v_sel,
	double const* const __restrict__ vecs, u64& v_r) {
	for (u64 i = 0; i < data_vocab_size; i++)
	{
		if (r[i] == 0)
			continue;
		r_sel[v_r] = r[i];

		// copy those words
#pragma omp parallel for
		for (u64 k = 0; k < word2vec_word_embedding_size; k++) {
			v_sel[v_r * word2vec_word_embedding_size + k] =
				vecs[i * word2vec_word_embedding_size + k];
		}
		v_r++;
	}
}


void SDDMM2(const u64 data_vocab_size, const u64 v_r, const u64 num_docs,
	double const* const __restrict__ K_transpose,
	double const* const __restrict__ u, double* w_csr) {
#pragma omp parallel for
	for (u64 i = 0; i < data_vocab_size; i++) {
		//can be parallel as well.
		for (u64 e = c_csr.row_ptr[i]; e < c_csr.row_ptr[i + 1]; e++) {
			u64 j = c_csr.col_inds[e];
			double sum = 0.;
			// #pragma vector
			for (u64 k = 0; k < v_r; k++) {
				sum += (K_transpose[i * v_r + k] * u[j * v_r + k]);
			}

			//[Vxnum_docs]
			w_csr[e] = ((c_csr.vals[e] / sum));
		}
	}
}
void print_matrix(double* X, u64 rows, u64 cols) {
	for (u64 i = 0; i < rows; i++) {

		for (u64 j = 0; j < cols; j++) {

			cout << X[i * cols + j] << "\n";

		}
	}
}

void print_csr(u64 num_rows, double* vals, u64* col_inds, u64* row_ptr) {
	for (u64 i = 0; i < num_rows; i++) {
		double sum = 0.;
		for (u64 e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
			u64 j = col_inds[e];
			//[Vxnum_docs]
			cout << i << "," << j << "," << vals[e] << endl;

		}
	}
}
/*************************** Utilitty Functions **************************/
void sinkhorn_wmd(const double* r, const double* vecs, double* WMD,
	const double lamda, const int max_iter, const u64 data_vocab_size,
	const u64 num_docs, const u64 word2vec_word_embedding_size) {

	// select non-zero entry


	// select non-zero word embedding
	/************************************************************************/

	select_non_empty_entries(data_vocab_size, r, r_sel, v_sel, vecs, v_r);
	/************************************************************************/
	// M = cdist(vecs[sel], vecs).astype(np.float64)
#ifdef VERBOSE
	cout << "volcabulary size in the input doc: " << v_r << endl;
#endif
	double* M;
	assert(posix_memalign((void**)&M, ALN, (v_r * data_vocab_size + 1) * sizeof(double)) == 0);
	//assert(M != NULL);

	//# K=exp(-lambda * M)
	//  K = np.exp(- M * lamb)
	double* K;
	assert(posix_memalign((void**)&K, ALN, (v_r * data_vocab_size + 1) * sizeof(double)) == 0);
	//assert(K != NULL);

	double* K_transpose;
	assert(posix_memalign((void**)&K_transpose, ALN,
		(v_r * data_vocab_size + 1) * sizeof(double)) == 0);
	//assert(K_transpose != NULL);

	double* x;
	assert(posix_memalign((void**)&x, ALN, (v_r * num_docs + 1) * sizeof(double)) == 0);
	//assert(x != NULL);

	// u=1/x
	double* u;
	assert(posix_memalign((void**)&u, ALN, (v_r * num_docs + 1) * sizeof(double)) == 0);
	//assert(u != NULL);

	double* K_over_r;
	assert(posix_memalign((void**)&K_over_r, ALN,
		(data_vocab_size * v_r + 1) * sizeof(double)) == 0);
	//assert(K_over_r != NULL);

	double* w_csr;
	assert(posix_memalign((void**)&w_csr, ALN, (c_csr.nnz + 1) * sizeof(double)) == 0);
	//assert(w_csr != NULL);

	// compute distance between word embeding
	// compute K
	// compute K.T
#pragma omp parallel for
	for (u64 i = 0; i < v_r; i++) {
		for (u64 j = 0; j < data_vocab_size; j++) {
			M[i * data_vocab_size + j] = euclidean_distance(
				&v_sel[i * word2vec_word_embedding_size],
				&vecs[j * word2vec_word_embedding_size]);
			K[i * data_vocab_size + j] = exp(
				-M[i * data_vocab_size + j] * lamda);
			K_transpose[j * v_r + i] = K[i * data_vocab_size + j];
		}
	}

	/************************************************************************/

	// compute initial x
#pragma omp parallel for
	for (u64 i = 0; i < v_r; i++) {
#pragma vector
		for (u64 j = 0; j < num_docs; j++) {
			x[i * num_docs + j] = 0.;
			u[i * num_docs + j] = v_r;
			// cout<<u[i * num_docs + j] <<" ";
		}
	}
	// compute k over r
	// K_over_r=(1 / r) * K
	// vector * matrix
#pragma omp parallel for
	for (u64 i = 0; i < v_r; i++) {
		double multiplier = 1.0 / r_sel[i];
#pragma vector
		for (u64 j = 0; j < data_vocab_size; j++) {
			K_over_r[i * data_vocab_size + j] = multiplier
				* K[i * data_vocab_size + j];
		}
	}

#pragma omp parallel for
	for (u64 i = 0; i < c_csr.nnz; i++) {
		w_csr[i] = 0.;
	}

	/***************************Main Loop *********************************/
	// # while x changes: x=diag(1./r)*K*(c_csr.*(1./(K’*(1./x))))
	//unsigned long long start = 0, end = 0;

	int iteration = 0;
	while (iteration < max_iter) {
		//start = getMilliCount();
		// SDDMM
#if 0
		print_matrix(K_transpose, data_vocab_size, v_r);
		print_matrix(K_over_r, v_r, data_vocab_size);
		print_matrix(u, v_r, num_docs);
		print_csr(c_csr.num_rows, c_csr.vals, c_csr.col_inds, c_csr.row_ptr);
#endif

		// w_csr = (K.T @ u)
		// v = c_csr.multiply(1 / w_csr)
		//SDDMM(data_vocab_size, v_r, num_docs, K_transpose, u, w_csr);
		SDDMM2(data_vocab_size, v_r, num_docs, K_transpose, u, w_csr);
#if 0
		print_csr(c_csr.num_rows, w_csr, c_csr.col_inds, c_csr.row_ptr);
#endif
		// convert the csr to csc
		// csr2csc(c_csr.num_rows, c_csr.num_cols, c_csr.nnz, c_csr.vals,
		// c_csr.col_inds, c_csr.row_ptr, w_csc.vals, w_csc.row_inds, w_csc.col_ptr);

		// x=(K_over_r@w)
		// u = 1/x;
		// SPDM
#pragma omp parallel for
		for (u64 i = 0; i < v_r; i++)
		{
			for (u64 k = 0; k < data_vocab_size; k++) {
				double multiplier = K_over_r[i * data_vocab_size + k];
				for (u64 e = c_csr.row_ptr[k]; e < c_csr.row_ptr[k + 1]; e++) {
					u64 j = c_csr.col_inds[e];
					x[i * num_docs + j] += multiplier * w_csr[e];
				}
			}
		}

#if 0
		print_matrix(x, v_r, num_docs);
#endif

#pragma omp parallel for
		for (u64 i = 0; i < v_r; i++) {
#pragma vector
			for (u64 j = 0; j < num_docs; j++) {
				//u[i * num_docs + j] = 1.0 / x[i * num_docs + j];
				u[j * v_r + i] = 1.0 / x[i * num_docs + j];
				x[i * num_docs + j] = 0.;
			}
		}

		iteration++;
		//end += getMilliSpan(start);
		//cout << "It=" << iteration << " |" << (double) end / 1000.0 << " s"
		//		<< endl;
	}

	//  u = 1.0 / x
	//  w_csr.T = (c_csr.multiply(1 / (K.T @ u)))T
	//SDDMM(data_vocab_size, v_r, num_docs, K_transpose, u, w_csr);
	SDDMM2(data_vocab_size, v_r, num_docs, K_transpose, u, w_csr);
#if 0
	print_csr(c_csr.num_rows, w_csr, c_csr.col_inds, c_csr.row_ptr);
#endif
	// return (u * ((K * M) @ w)).sum(axis=0)
	// w_csr.T @ (K*M)T
	// Dense * Sparse
#pragma omp parallel for
	for (u64 i = 0; i < v_r; i++) {
		for (u64 k = 0; k < data_vocab_size; k++) {
			double multiplier = K[i * data_vocab_size + k]
				* M[i * data_vocab_size + k];

			for (u64 e = c_csr.row_ptr[k]; e < c_csr.row_ptr[k + 1]; e++) {
				u64 j = c_csr.col_inds[e];
				// using x for this purpose!
				x[i * num_docs + j] += multiplier * w_csr[e];
			}
		}
	}

#pragma omp parallel for
	for (u64 j = 0; j < num_docs; j++)
		WMD[j] = 0.;

#pragma omp parallel for
	for (u64 j = 0; j < num_docs; j++) {
		double wmd_j = 0.0;
		for (u64 i = 0; i < v_r; i++) {
			//x[i * num_docs + j] = u[i * num_docs + j] * x[i * num_docs + j];
			x[i * num_docs + j] = u[j * v_r + i] * x[i * num_docs + j];
			wmd_j += x[i * num_docs + j];
		}
		WMD[j] = wmd_j;
	}

	// free all allocated memory
	free(x);
	free(u);
	free(w_csr);
	free(K);
	free(M);
	free(K_transpose);
	free(K_over_r);
	free(v_sel);
	free(r_sel);
}

int main(int argc, char* argv[]) {
	const char* mat_filename = "./data/mat.mtx";

	cout << "Reading c_csr matrix csr: " << mat_filename << endl;
	ifstream mat_file(mat_filename);
	if (!mat_file.is_open()) {
		cout << "Could not open the file:" << mat_filename << " exiting\n";
		exit(0);
	}

	string line;
	mat_file >> c_csr.num_rows;
	mat_file >> c_csr.num_cols;
	mat_file >> c_csr.nnz;
    
    if (c_csr.num_rows <= 0 || c_csr.num_cols <= 0 || c_csr.nnz <= 0)
	{
		cout << "CSR contains invalid input\n";
		exit(0);
	}

#ifdef VERBOSE
	cout << "Running: " << argv[0] << ", V:" << c_csr.num_rows
		<< ", docs:" << c_csr.num_cols << ", occurances:" << c_csr.nnz << endl;
#endif
	// allocate memory for the graph
	assert(posix_memalign((void**)&c_csr.row_ptr, ALN,
		(c_csr.num_rows + 2) * sizeof(u64)) == 0);
	assert(c_csr.row_ptr != NULL);

	assert(posix_memalign((void**)&c_csr.col_inds, ALN, (c_csr.nnz + 1) * sizeof(u64)) == 0);
	assert(c_csr.col_inds != NULL);

	assert(posix_memalign((void**)&c_csr.vals, ALN, (c_csr.nnz + 1) * sizeof(double)) == 0);
	assert(c_csr.vals != NULL);

	u64* degree;
	assert(posix_memalign((void**)&degree, ALN, (c_csr.num_rows + 2) * sizeof(u64)) == 0);
	assert(degree != NULL);

	connector_t* edgeList;
	assert(posix_memalign((void**)&edgeList, ALN, (c_csr.nnz + 1) * sizeof(connector_t)) == 0);
	assert(edgeList != NULL);

#pragma omp parallel for
	for (u64 i = 0; i < c_csr.num_rows; i++) {
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
        
        if (word1 <= 0 || word2 <= 0 || w <= 0 || word1 > c_csr.num_rows || word2 > c_csr.num_cols)
		{
			cout << "CSR contains invalid input\n";
			exit(0);
		}

		edgeList[i].u = word1 - 1;
		edgeList[i].v = word2 - 1;
		edgeList[i].edge_weight = w;

		degree[word1 - 1]++;
		// cout<<edgeList[i].u<<" " <<edgeList[i].v<<"
		// "<<edgeList[i].edge_weight<<endl;
		i++;
	}

	mat_file.close();
    assert(i==c_csr.nnz);
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
		c_csr.vals[k] = (double)edgeList[i].edge_weight;
		degree[word1]++;
	}

	free(edgeList);
	free(degree);

	//posix_memalign((void **)&w_csc.col_ptr, ALN,
	//               (c_csr.num_cols + 1) * sizeof(u64));
	//assert(w_csc.col_ptr != NULL);

	//posix_memalign((void **)&w_csc.row_inds, ALN, (c_csr.nnz) * sizeof(u64));
	//assert(w_csc.row_inds != NULL);

	//posix_memalign((void **)&w_csc.vals, ALN, (c_csr.nnz) * sizeof(double));
	//assert(w_csc.vals != NULL);
	// Now read from the r.out file: this is the word frequency in the input file.

	const char* r_filename = "./data/r.out";
	cout << "reading r (sparse vector): " << r_filename << endl;
	ifstream r_file(r_filename);

	assert(posix_memalign((void**)&r_arr, ALN, (c_csr.num_rows + 1) * sizeof(double)) == 0);
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
        if (val < 0)
		{
			cout << "Source contains invalid input\n";
			exit(0);
		}
		i++;
	}
	cout << "r_arr length:" << i - 1 << endl;
	r_file.close();

	//////////////////////////////////////////////////////////////////
	const char* v_filename = "./data/vecs.out";

	ifstream v_file(v_filename);

	assert(posix_memalign((void**)&vecs, ALN,
		(c_csr.num_rows * word2vec_word_embedding_size + 1) * sizeof(double)) == 0);
	assert(vecs != NULL);

	if (!v_file.is_open()) {
		cout << "Could not open the file:" << v_filename << " exiting\n";
		exit(0);
	}
	i = 0;
	for (u64 j = 0; j < c_csr.num_rows && !v_file.eof(); j++) {
		getline(v_file, line);
		stringstream linestream(line);
		string data;
		u64 l = 0;
		while (std
			::getline(linestream, data, ' ')) {
			vecs[j * word2vec_word_embedding_size + l] = stod(data);
			l++;
		}
		if (l != word2vec_word_embedding_size)
		{
			cout << "Did not read enough embedding: " << "expecting"<< word2vec_word_embedding_size<< " read "<<l << " exiting\n";
			exit(0);
		}
		i++;

	}
	v_file.close();

	if (c_csr.num_rows != i)
	{
		cout << "Dimensions do not match " << v_filename << " exiting\n";
		exit(0);
	}
	i = 0;


	assert(posix_memalign((void**)&r_sel, ALN, (c_csr.num_rows + 1) * sizeof(double)) == 0);
	assert(r_sel != NULL);


	assert(posix_memalign((void**)&v_sel, ALN,
		(c_csr.num_rows * word2vec_word_embedding_size + 1) * sizeof(double)) == 0);
	assert(v_sel != NULL);
	double* WMD;
	assert(posix_memalign((void**)&WMD, ALN, (c_csr.num_cols + 1) * sizeof(double)) == 0);
	assert(WMD != NULL);

	int nmtp_threads;
#pragma omp parallel
	{
		nmtp_threads = omp_get_num_threads();
	}
	cout << "num_threads " << nmtp_threads << endl;
	// Main function.
	unsigned long long start = 0, end = 0;
	start = getMilliCount();

	sinkhorn_wmd(r_arr, vecs, WMD, lamda, max_iter, c_csr.num_rows,
		c_csr.num_cols, word2vec_word_embedding_size);

	end = getMilliSpan(start);
	cout << "volcabulary size in the input doc: " << v_r << endl;
	cout << "elapsed=" << (double)end / 1000.0 << endl;
	// write output files in scores file
	ofstream outfile;
	outfile.open("scores", std::ofstream::out);
	for (u64 j = 0; j < c_csr.num_cols; j++)
		outfile << std::setprecision(9) << WMD[j] << endl;
	outfile.close();

	//print_csr(c_csr.num_rows, c_csr.vals, c_csr.col_inds, c_csr.row_ptr);

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
	return 0;
}
