#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;

enum wwama_tensor_status {
    WWAMA_TENSOR_OK = 0,
    WWAMA_TENSOR_INVALID_ARGUMENT = 1,
    WWAMA_TENSOR_NOT_FOUND = 2,
    WWAMA_TENSOR_OUT_OF_BOUNDS = 3,
    WWAMA_TENSOR_UNAVAILABLE = 4,
};

struct wwama_tensor_descriptor {
    const char * name;
    const char * type_name;
    const char * backend_name;
    int32_t type_id;
    int32_t n_dims;
    int64_t ne[4];
    size_t nb[4];
    size_t nbytes;
};

size_t wwama_tensor_count(const struct llama_model * model);

int32_t wwama_tensor_descriptor_at(
    const struct llama_model * model,
    size_t index,
    struct wwama_tensor_descriptor * descriptor);

int32_t wwama_tensor_descriptor_named(
    const struct llama_model * model,
    const char * name,
    struct wwama_tensor_descriptor * descriptor);

int32_t wwama_tensor_read(
    const struct llama_model * model,
    const char * name,
    size_t offset,
    void * destination,
    size_t size);

int32_t wwama_tensor_write(
    struct llama_model * model,
    const char * name,
    size_t offset,
    const void * source,
    size_t size);

#ifdef __cplusplus
}
#endif
