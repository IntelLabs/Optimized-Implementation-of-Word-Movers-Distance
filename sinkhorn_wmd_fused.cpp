#include "common.h"

using namespace std;


double* r_sel;
double* v_sel;

u64 v_r = 0;
/******************************* Global Variables ************************/

int max_iter = 15;
puma_csr_double_t c_csr;
//puma_csc_double_t w_csc;

double* r_arr;
double* vecs;
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

/*************************** Utilitty Functions **************************/
REAL inline euclidean_distance(double const* const u, double const* const v) {
	REAL s = 0.0;
#pragma unroll
	for (u64 i = 0; i < word2vec_word_embedding_size; ++i) {
		const REAL d = u[i] - v[i];
		s += (d * d);
	}
	return sqrt(s);
}

void inline select_non_empty_entries(u64 data_vocab_size, double const* const __restrict__ r,
	double* __restrict__ r_sel, double* __restrict__ v_sel,
	double const* const __restrict__ vecs, u64& v_r) {
	for (u64 i = 0; i < data_vocab_size; i++)
	{
		if (r[i] == 0)
			continue;
		r_sel[v_r] = 1.0 / r[i];

		// copy those words
		double* v_sel_ptr = v_sel + v_r * word2vec_word_embedding_size;
		const double* vecs_ptr = vecs + i * word2vec_word_embedding_size;
#pragma omp parallel for
		for (u64 k = 0; k < word2vec_word_embedding_size; k++) {
			v_sel_ptr[k] = vecs_ptr[k];
		}
		v_r++;
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

static inline u64 p_binary_search(
	u64 const* const __restrict__ weights,
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
		}
		else {
			right = mid;
		}
	}

	return right;
}

/*************************** Utilitty Functions **************************/
void inline sinkhorn_wmd(const double* r, const double* vecs, REAL* WMD,
	const REAL lamda, const int max_iter, const u64 data_vocab_size,
	const u64 num_docs, const u64 word2vec_word_embedding_size) {

	// select non-zero word embedding
	/************************************************************************/

	select_non_empty_entries(data_vocab_size, r, r_sel, v_sel, vecs, v_r);



	/************************************************************************/
	// M = cdist(vecs[sel], vecs).astype(np.float64)

#ifdef VERBOSE
	cout << "volcabulary size in the input doc: " << v_r << endl;
	cout << "num docs: " << num_docs << endl;
#endif
	REAL* M;
	assert(posix_memalign((void**)&M, ALN, (v_r * data_vocab_size + 1) * sizeof(REAL)) == 0);
	//assert(M != NULL);

	//# K=exp(-lambda * M)
	//  K = np.exp(- M * lamb)
	REAL* K;
	assert(posix_memalign((void**)&K, ALN, (v_r * data_vocab_size + 1) * sizeof(REAL)) == 0);
	//assert(K != NULL);


	REAL* x;
	assert(posix_memalign((void**)&x, ALN, (v_r * num_docs + 1) * sizeof(REAL)) == 0);
	//assert(x != NULL);

	// u=1/x
	REAL* u;
	assert(posix_memalign((void**)&u, ALN, (v_r * num_docs + 1) * sizeof(REAL)) == 0);
	//assert(u != NULL);

	REAL* K_over_r;
	assert(posix_memalign((void**)&K_over_r, ALN,
		(data_vocab_size * v_r + 1) * sizeof(REAL)) == 0);
	//assert(K_over_r != NULL);

#pragma omp parallel
	{
#pragma omp barrier
	}
	//unsigned long long start = 0, end = 0;
	//start = getMilliCount();
	u64 const* csr_row_ptr = c_csr.row_ptr;
	u64 const* csr_cols = c_csr.col_inds;
	double const* csr_vals = c_csr.vals;
	u64 total_items_csr = c_csr.nnz;
	u64 csr_item_per_mtp = (total_items_csr + nmtp_threads - 1)
		/ nmtp_threads;
	u64* mtps_start_item_csr, * mtps_end_item_csr, * row_id_start;
	mtps_start_item_csr = new u64[nmtp_threads];
	mtps_end_item_csr = new u64[nmtp_threads];
	row_id_start = new u64[nmtp_threads];
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

			REAL* M_ptr = M + i_offset;
			REAL* K_ptr = K + i_offset;
			REAL* K_over_r_ptr = K_over_r + i_offset;

			double const* const v_prt = vecs + j * word2vec_word_embedding_size;
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

				REAL* KT_ptr = K + row_id * v_r;
				REAL* u_ptr = u + j * v_r;
				for (u64 k = 0; k < v_r; k++) {
					sum += (KT_ptr[k] * u_ptr[k]);
				}

				//[Vxnum_docs]
				REAL const output = ((val / sum));

				REAL* x_ptr = x + j * v_r;
				REAL* K_r = (K_over_r + row_id * v_r);
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
			REAL* KT_prt = (K + row_id * v_r);
			REAL* u_ptr = (u + j * v_r);
#pragma unroll
			for (u64 k = 0; k < v_r; k++) {
				sum += (KT_prt[k] * u_ptr[k]);
			}

			//[Vxnum_docs]
			const REAL output = ((val / sum));
			//cout<<output<<endl;
			REAL* x_ptr = x + j * v_r;
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
		REAL* u_ptr = u + j * v_r;
		REAL* x_ptr = x + j * v_r;
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
	//end = getMilliSpan(start);
	//cout << "elapsed=" << (double)end / 1000.0 << endl;
	// free all allocated memory
	free(x);
	free(u);
	free(K);
	free(M);
	free(K_over_r);
	free(v_sel);
	free(r_sel);
	delete[] mtps_start_item_csr;
	delete[] mtps_end_item_csr;
	delete[] row_id_start;
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

	if (c_csr.num_rows == 0 || c_csr.num_cols == 0 || c_csr.nnz == 0)
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
		c_csr.vals[k] = (double)edgeList[i].edge_weight;
		degree[word1]++;
	}

	free(edgeList);

	//////////////////////////////////////////////////////////////////
	const char* v_filename = "./data/vecs.out";
	cout << "reading vecs (sparse vector): " << v_filename << endl;
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
		for (u64 l = 0; l < word2vec_word_embedding_size; l++) {
			std
				::getline(linestream, data, ','); // read up-to the
			vecs[j * word2vec_word_embedding_size + l] = stod(data);
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
		//cout<<r_arr[i]<<endl;
		i++;
	}

	r_file.close();

	if (--i != c_csr.num_rows)
	{
		cout << "Dimension mismatch in:" << r_filename << " exiting\n";
		exit(0);
	}

	// select non-zero entry
	assert(posix_memalign((void**)&r_sel, ALN, (c_csr.num_rows + 1) * sizeof(double)) == 0);
	assert(r_sel != NULL);

	assert(posix_memalign((void**)&v_sel, ALN,
		(c_csr.num_rows * word2vec_word_embedding_size + 1) * sizeof(double)) == 0);
	assert(v_sel != NULL);

	REAL* WMD;
	assert(posix_memalign((void**)&WMD, ALN, (c_csr.num_cols + 1) * sizeof(REAL)) == 0);
	assert(WMD != NULL);

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
	for (u64 j = 0; j < c_csr.num_cols; j++) {
		outfile << std
			::setprecision(9) << WMD[j] << endl;
		//cout<<std::setprecision(9) << WMD[j] << endl;
	}
	outfile.close();

	// free allocated memory
	free(WMD);
	free(vecs);
	free(r_arr);

	free(c_csr.col_inds);
	free(c_csr.row_ptr);
	free(c_csr.vals);
	free(degree);

	//free(w_csc.row_inds);
	//free(w_csc.col_ptr);
	//free(w_csc.vals);
	return 0;
}
