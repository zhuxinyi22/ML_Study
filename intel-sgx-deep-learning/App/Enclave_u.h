#ifndef ENCLAVE_U_H__
#define ENCLAVE_U_H__

#include <stdint.h>
#include <wchar.h>
#include <stddef.h>
#include <string.h>
#include "sgx_edger8r.h" /* for sgx_status_t etc. */


#include <stdlib.h> /* for size_t */

#define SGX_CAST(type, item) ((type)(item))

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OCALL_PRINT_STRING_DEFINED__
#define OCALL_PRINT_STRING_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_print_string, (const char* str));
#endif
#ifndef OCALL_START_MEASURING_TRAINING_DEFINED__
#define OCALL_START_MEASURING_TRAINING_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_start_measuring_training, (int sub_time_index, int repetitions));
#endif
#ifndef OCALL_END_MEASURING_TRAINING_DEFINED__
#define OCALL_END_MEASURING_TRAINING_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_end_measuring_training, (int sub_time_index, int repetitions));
#endif
#ifndef OCALL_SPAWN_THREADS_DEFINED__
#define OCALL_SPAWN_THREADS_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_spawn_threads, (int n));
#endif
#ifndef OCALL_PUSH_WEIGHTS_DEFINED__
#define OCALL_PUSH_WEIGHTS_DEFINED__
void SGX_UBRIDGE(SGX_NOCONVENTION, ocall_push_weights, (const char* ptr, size_t size, size_t nmemb));
#endif

sgx_status_t ecall_train_network(sgx_enclave_id_t eid, char* train_file, int size_train_file, int num_threads);
sgx_status_t ecall_test_network(sgx_enclave_id_t eid, char* test_file, int size_test_file, int num_threads);
sgx_status_t ecall_thread_enter_enclave_waiting(sgx_enclave_id_t eid, int thread_id);
sgx_status_t ecall_build_network(sgx_enclave_id_t eid, char* file_string, size_t len_string, char* weights, size_t size_weights);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
