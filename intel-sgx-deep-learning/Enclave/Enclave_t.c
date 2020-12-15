#include "Enclave_t.h"

#include "sgx_trts.h" /* for sgx_ocalloc, sgx_is_outside_enclave */
#include "sgx_lfence.h" /* for sgx_lfence */

#include <errno.h>
#include <mbusafecrt.h> /* for memcpy_s etc */
#include <stdlib.h> /* for malloc/free etc */

#define CHECK_REF_POINTER(ptr, siz) do {	\
	if (!(ptr) || ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_UNIQUE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_outside_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define CHECK_ENCLAVE_POINTER(ptr, siz) do {	\
	if ((ptr) && ! sgx_is_within_enclave((ptr), (siz)))	\
		return SGX_ERROR_INVALID_PARAMETER;\
} while (0)

#define ADD_ASSIGN_OVERFLOW(a, b) (	\
	((a) += (b)) < (b)	\
)


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

static sgx_status_t SGX_CDECL sgx_ecall_train_network(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_train_network_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_train_network_t* ms = SGX_CAST(ms_ecall_train_network_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_train_file = ms->ms_train_file;
	int _tmp_size_train_file = ms->ms_size_train_file;
	size_t _len_train_file = _tmp_size_train_file * sizeof(char);
	char* _in_train_file = NULL;

	if (sizeof(*_tmp_train_file) != 0 &&
		(size_t)_tmp_size_train_file > (SIZE_MAX / sizeof(*_tmp_train_file))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_train_file, _len_train_file);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_train_file != NULL && _len_train_file != 0) {
		if ( _len_train_file % sizeof(*_tmp_train_file) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_train_file = (char*)malloc(_len_train_file);
		if (_in_train_file == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_train_file, _len_train_file, _tmp_train_file, _len_train_file)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ecall_train_network(_in_train_file, _tmp_size_train_file, ms->ms_num_threads);

err:
	if (_in_train_file) free(_in_train_file);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_test_network(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_test_network_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_test_network_t* ms = SGX_CAST(ms_ecall_test_network_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_test_file = ms->ms_test_file;
	int _tmp_size_test_file = ms->ms_size_test_file;
	size_t _len_test_file = _tmp_size_test_file * sizeof(char);
	char* _in_test_file = NULL;

	if (sizeof(*_tmp_test_file) != 0 &&
		(size_t)_tmp_size_test_file > (SIZE_MAX / sizeof(*_tmp_test_file))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_test_file, _len_test_file);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_test_file != NULL && _len_test_file != 0) {
		if ( _len_test_file % sizeof(*_tmp_test_file) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_test_file = (char*)malloc(_len_test_file);
		if (_in_test_file == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_test_file, _len_test_file, _tmp_test_file, _len_test_file)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ecall_test_network(_in_test_file, _tmp_size_test_file, ms->ms_num_threads);

err:
	if (_in_test_file) free(_in_test_file);
	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_thread_enter_enclave_waiting(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_thread_enter_enclave_waiting_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_thread_enter_enclave_waiting_t* ms = SGX_CAST(ms_ecall_thread_enter_enclave_waiting_t*, pms);
	sgx_status_t status = SGX_SUCCESS;



	ecall_thread_enter_enclave_waiting(ms->ms_thread_id);


	return status;
}

static sgx_status_t SGX_CDECL sgx_ecall_build_network(void* pms)
{
	CHECK_REF_POINTER(pms, sizeof(ms_ecall_build_network_t));
	//
	// fence after pointer checks
	//
	sgx_lfence();
	ms_ecall_build_network_t* ms = SGX_CAST(ms_ecall_build_network_t*, pms);
	sgx_status_t status = SGX_SUCCESS;
	char* _tmp_file_string = ms->ms_file_string;
	size_t _tmp_len_string = ms->ms_len_string;
	size_t _len_file_string = _tmp_len_string * sizeof(char);
	char* _in_file_string = NULL;
	char* _tmp_weights = ms->ms_weights;
	size_t _tmp_size_weights = ms->ms_size_weights;
	size_t _len_weights = _tmp_size_weights * sizeof(char);
	char* _in_weights = NULL;

	if (sizeof(*_tmp_file_string) != 0 &&
		(size_t)_tmp_len_string > (SIZE_MAX / sizeof(*_tmp_file_string))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	if (sizeof(*_tmp_weights) != 0 &&
		(size_t)_tmp_size_weights > (SIZE_MAX / sizeof(*_tmp_weights))) {
		return SGX_ERROR_INVALID_PARAMETER;
	}

	CHECK_UNIQUE_POINTER(_tmp_file_string, _len_file_string);
	CHECK_UNIQUE_POINTER(_tmp_weights, _len_weights);

	//
	// fence after pointer checks
	//
	sgx_lfence();

	if (_tmp_file_string != NULL && _len_file_string != 0) {
		if ( _len_file_string % sizeof(*_tmp_file_string) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_file_string = (char*)malloc(_len_file_string);
		if (_in_file_string == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_file_string, _len_file_string, _tmp_file_string, _len_file_string)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}
	if (_tmp_weights != NULL && _len_weights != 0) {
		if ( _len_weights % sizeof(*_tmp_weights) != 0)
		{
			status = SGX_ERROR_INVALID_PARAMETER;
			goto err;
		}
		_in_weights = (char*)malloc(_len_weights);
		if (_in_weights == NULL) {
			status = SGX_ERROR_OUT_OF_MEMORY;
			goto err;
		}

		if (memcpy_s(_in_weights, _len_weights, _tmp_weights, _len_weights)) {
			status = SGX_ERROR_UNEXPECTED;
			goto err;
		}

	}

	ecall_build_network(_in_file_string, _tmp_len_string, _in_weights, _tmp_size_weights);

err:
	if (_in_file_string) free(_in_file_string);
	if (_in_weights) free(_in_weights);
	return status;
}

SGX_EXTERNC const struct {
	size_t nr_ecall;
	struct {void* ecall_addr; uint8_t is_priv; uint8_t is_switchless;} ecall_table[4];
} g_ecall_table = {
	4,
	{
		{(void*)(uintptr_t)sgx_ecall_train_network, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_test_network, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_thread_enter_enclave_waiting, 0, 0},
		{(void*)(uintptr_t)sgx_ecall_build_network, 0, 0},
	}
};

SGX_EXTERNC const struct {
	size_t nr_ocall;
	uint8_t entry_table[5][4];
} g_dyn_entry_table = {
	5,
	{
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
		{0, 0, 0, 0, },
	}
};


sgx_status_t SGX_CDECL ocall_print_string(const char* str)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_str = str ? strlen(str) + 1 : 0;

	ms_ocall_print_string_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_print_string_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(str, _len_str);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (str != NULL) ? _len_str : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_print_string_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_print_string_t));
	ocalloc_size -= sizeof(ms_ocall_print_string_t);

	if (str != NULL) {
		ms->ms_str = (const char*)__tmp;
		if (_len_str % sizeof(*str) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_s(__tmp, ocalloc_size, str, _len_str)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_str);
		ocalloc_size -= _len_str;
	} else {
		ms->ms_str = NULL;
	}
	
	status = sgx_ocall(0, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_start_measuring_training(int sub_time_index, int repetitions)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_start_measuring_training_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_start_measuring_training_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_start_measuring_training_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_start_measuring_training_t));
	ocalloc_size -= sizeof(ms_ocall_start_measuring_training_t);

	ms->ms_sub_time_index = sub_time_index;
	ms->ms_repetitions = repetitions;
	status = sgx_ocall(1, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_end_measuring_training(int sub_time_index, int repetitions)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_end_measuring_training_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_end_measuring_training_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_end_measuring_training_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_end_measuring_training_t));
	ocalloc_size -= sizeof(ms_ocall_end_measuring_training_t);

	ms->ms_sub_time_index = sub_time_index;
	ms->ms_repetitions = repetitions;
	status = sgx_ocall(2, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_spawn_threads(int n)
{
	sgx_status_t status = SGX_SUCCESS;

	ms_ocall_spawn_threads_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_spawn_threads_t);
	void *__tmp = NULL;


	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_spawn_threads_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_spawn_threads_t));
	ocalloc_size -= sizeof(ms_ocall_spawn_threads_t);

	ms->ms_n = n;
	status = sgx_ocall(3, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

sgx_status_t SGX_CDECL ocall_push_weights(const char* ptr, size_t size, size_t nmemb)
{
	sgx_status_t status = SGX_SUCCESS;
	size_t _len_ptr = nmemb * size;

	ms_ocall_push_weights_t* ms = NULL;
	size_t ocalloc_size = sizeof(ms_ocall_push_weights_t);
	void *__tmp = NULL;


	CHECK_ENCLAVE_POINTER(ptr, _len_ptr);

	if (ADD_ASSIGN_OVERFLOW(ocalloc_size, (ptr != NULL) ? _len_ptr : 0))
		return SGX_ERROR_INVALID_PARAMETER;

	__tmp = sgx_ocalloc(ocalloc_size);
	if (__tmp == NULL) {
		sgx_ocfree();
		return SGX_ERROR_UNEXPECTED;
	}
	ms = (ms_ocall_push_weights_t*)__tmp;
	__tmp = (void *)((size_t)__tmp + sizeof(ms_ocall_push_weights_t));
	ocalloc_size -= sizeof(ms_ocall_push_weights_t);

	if (ptr != NULL) {
		ms->ms_ptr = (const char*)__tmp;
		if (_len_ptr % sizeof(*ptr) != 0) {
			sgx_ocfree();
			return SGX_ERROR_INVALID_PARAMETER;
		}
		if (memcpy_s(__tmp, ocalloc_size, ptr, _len_ptr)) {
			sgx_ocfree();
			return SGX_ERROR_UNEXPECTED;
		}
		__tmp = (void *)((size_t)__tmp + _len_ptr);
		ocalloc_size -= _len_ptr;
	} else {
		ms->ms_ptr = NULL;
	}
	
	ms->ms_size = size;
	ms->ms_nmemb = nmemb;
	status = sgx_ocall(4, ms);

	if (status == SGX_SUCCESS) {
	}
	sgx_ocfree();
	return status;
}

