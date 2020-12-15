#include "Enclave_u.h"
#include <errno.h>

typedef struct ms_ecall_train_network_t {
	char* ms_train_file;
	int ms_size_train_file;
	int ms_num_threads;
} ms_ecall_train_network_t;

typedef struct ms_ecall_test_network_t {
	char* ms_test_file;
	int ms_size_test_file;
	int ms_num_threads;
} ms_ecall_test_network_t;

typedef struct ms_ecall_thread_enter_enclave_waiting_t {
	int ms_thread_id;
} ms_ecall_thread_enter_enclave_waiting_t;

typedef struct ms_ecall_build_network_t {
	char* ms_file_string;
	size_t ms_len_string;
	char* ms_weights;
	size_t ms_size_weights;
} ms_ecall_build_network_t;

typedef struct ms_ocall_print_string_t {
	const char* ms_str;
} ms_ocall_print_string_t;

typedef struct ms_ocall_start_measuring_training_t {
	int ms_sub_time_index;
	int ms_repetitions;
} ms_ocall_start_measuring_training_t;

typedef struct ms_ocall_end_measuring_training_t {
	int ms_sub_time_index;
	int ms_repetitions;
} ms_ocall_end_measuring_training_t;

typedef struct ms_ocall_spawn_threads_t {
	int ms_n;
} ms_ocall_spawn_threads_t;

typedef struct ms_ocall_push_weights_t {
	const char* ms_ptr;
	size_t ms_size;
	size_t ms_nmemb;
} ms_ocall_push_weights_t;

static sgx_status_t SGX_CDECL Enclave_ocall_print_string(void* pms)
{
	ms_ocall_print_string_t* ms = SGX_CAST(ms_ocall_print_string_t*, pms);
	ocall_print_string(ms->ms_str);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_start_measuring_training(void* pms)
{
	ms_ocall_start_measuring_training_t* ms = SGX_CAST(ms_ocall_start_measuring_training_t*, pms);
	ocall_start_measuring_training(ms->ms_sub_time_index, ms->ms_repetitions);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_end_measuring_training(void* pms)
{
	ms_ocall_end_measuring_training_t* ms = SGX_CAST(ms_ocall_end_measuring_training_t*, pms);
	ocall_end_measuring_training(ms->ms_sub_time_index, ms->ms_repetitions);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_spawn_threads(void* pms)
{
	ms_ocall_spawn_threads_t* ms = SGX_CAST(ms_ocall_spawn_threads_t*, pms);
	ocall_spawn_threads(ms->ms_n);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL Enclave_ocall_push_weights(void* pms)
{
	ms_ocall_push_weights_t* ms = SGX_CAST(ms_ocall_push_weights_t*, pms);
	ocall_push_weights(ms->ms_ptr, ms->ms_size, ms->ms_nmemb);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * table[5];
} ocall_table_Enclave = {
	5,
	{
		(void*)Enclave_ocall_print_string,
		(void*)Enclave_ocall_start_measuring_training,
		(void*)Enclave_ocall_end_measuring_training,
		(void*)Enclave_ocall_spawn_threads,
		(void*)Enclave_ocall_push_weights,
	}
};
sgx_status_t ecall_train_network(sgx_enclave_id_t eid, char* train_file, int size_train_file, int num_threads)
{
	sgx_status_t status;
	ms_ecall_train_network_t ms;
	ms.ms_train_file = train_file;
	ms.ms_size_train_file = size_train_file;
	ms.ms_num_threads = num_threads;
	status = sgx_ecall(eid, 0, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_test_network(sgx_enclave_id_t eid, char* test_file, int size_test_file, int num_threads)
{
	sgx_status_t status;
	ms_ecall_test_network_t ms;
	ms.ms_test_file = test_file;
	ms.ms_size_test_file = size_test_file;
	ms.ms_num_threads = num_threads;
	status = sgx_ecall(eid, 1, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_thread_enter_enclave_waiting(sgx_enclave_id_t eid, int thread_id)
{
	sgx_status_t status;
	ms_ecall_thread_enter_enclave_waiting_t ms;
	ms.ms_thread_id = thread_id;
	status = sgx_ecall(eid, 2, &ocall_table_Enclave, &ms);
	return status;
}

sgx_status_t ecall_build_network(sgx_enclave_id_t eid, char* file_string, size_t len_string, char* weights, size_t size_weights)
{
	sgx_status_t status;
	ms_ecall_build_network_t ms;
	ms.ms_file_string = file_string;
	ms.ms_len_string = len_string;
	ms.ms_weights = weights;
	ms.ms_size_weights = size_weights;
	status = sgx_ecall(eid, 3, &ocall_table_Enclave, &ms);
	return status;
}

