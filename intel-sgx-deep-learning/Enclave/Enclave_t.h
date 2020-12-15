#ifndef ENCLAVE_T_H__
#define ENCLAVE_T_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include "sgx_edger8r.h" /* for sgx_ocall etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

void ecall_train_network(char* train_file, int size_train_file, int num_threads);
void ecall_test_network(char* test_file, int size_test_file, int num_threads);
void ecall_thread_enter_enclave_waiting(int thread_id);
void ecall_build_network(char* file_string, size_t len_string, char* weights, size_t size_weights);

sgx_status_t SGX_CDECL ocall_print_string(const char* str);
sgx_status_t SGX_CDECL ocall_start_measuring_training(int sub_time_index, int repetitions);
sgx_status_t SGX_CDECL ocall_end_measuring_training(int sub_time_index, int repetitions);
sgx_status_t SGX_CDECL ocall_spawn_threads(int n);
sgx_status_t SGX_CDECL ocall_push_weights(const char* ptr, size_t size, size_t nmemb);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
