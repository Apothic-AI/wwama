#include "wwama_tensor_bridge.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-model.h"

#include <cstring>
#include <limits>

namespace {

ggml_tensor * find_tensor(const llama_model * model, const char * name) {
    if (model == nullptr || name == nullptr) {
        return nullptr;
    }
    const auto & tensors = llama_internal_get_tensor_map(model);
    for (const auto & entry : tensors) {
        if (entry.first == name) {
            return entry.second;
        }
    }
    return nullptr;
}

int32_t fill_descriptor(const std::pair<std::string, ggml_tensor *> & entry,
                        wwama_tensor_descriptor * descriptor) {
    if (descriptor == nullptr || entry.second == nullptr) {
        return WWAMA_TENSOR_INVALID_ARGUMENT;
    }
    const ggml_tensor * tensor = entry.second;
    descriptor->name = entry.first.c_str();
    descriptor->type_name = ggml_type_name(tensor->type);
    descriptor->backend_name = tensor->buffer == nullptr
        ? nullptr
        : ggml_backend_buffer_name(tensor->buffer);
    descriptor->type_id = static_cast<int32_t>(tensor->type);
    descriptor->n_dims = ggml_n_dims(tensor);
    std::memcpy(descriptor->ne, tensor->ne, sizeof(descriptor->ne));
    std::memcpy(descriptor->nb, tensor->nb, sizeof(descriptor->nb));
    descriptor->nbytes = ggml_nbytes(tensor);
    return WWAMA_TENSOR_OK;
}

bool transfer_in_bounds(const ggml_tensor * tensor, size_t offset, size_t size) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return false;
    }
    const size_t nbytes = ggml_nbytes(tensor);
    return offset <= nbytes && size <= nbytes - offset;
}

} // namespace

extern "C" size_t wwama_tensor_count(const llama_model * model) {
    return model == nullptr ? 0 : llama_internal_get_tensor_map(model).size();
}

extern "C" int32_t wwama_tensor_descriptor_at(
    const llama_model * model,
    size_t index,
    wwama_tensor_descriptor * descriptor) {
    if (model == nullptr || descriptor == nullptr) {
        return WWAMA_TENSOR_INVALID_ARGUMENT;
    }
    const auto & tensors = llama_internal_get_tensor_map(model);
    if (index >= tensors.size()) {
        return WWAMA_TENSOR_OUT_OF_BOUNDS;
    }
    return fill_descriptor(tensors[index], descriptor);
}

extern "C" int32_t wwama_tensor_descriptor_named(
    const llama_model * model,
    const char * name,
    wwama_tensor_descriptor * descriptor) {
    if (model == nullptr || name == nullptr || descriptor == nullptr) {
        return WWAMA_TENSOR_INVALID_ARGUMENT;
    }
    const auto & tensors = llama_internal_get_tensor_map(model);
    for (const auto & entry : tensors) {
        if (entry.first == name) {
            return fill_descriptor(entry, descriptor);
        }
    }
    return WWAMA_TENSOR_NOT_FOUND;
}

extern "C" int32_t wwama_tensor_read(
    const llama_model * model,
    const char * name,
    size_t offset,
    void * destination,
    size_t size) {
    if (model == nullptr || name == nullptr || (destination == nullptr && size != 0)) {
        return WWAMA_TENSOR_INVALID_ARGUMENT;
    }
    const ggml_tensor * tensor = find_tensor(model, name);
    if (tensor == nullptr) {
        return WWAMA_TENSOR_NOT_FOUND;
    }
    if (!transfer_in_bounds(tensor, offset, size)) {
        return tensor->buffer == nullptr ? WWAMA_TENSOR_UNAVAILABLE : WWAMA_TENSOR_OUT_OF_BOUNDS;
    }
    if (size != 0) {
        ggml_backend_tensor_get(tensor, destination, offset, size);
    }
    return WWAMA_TENSOR_OK;
}

extern "C" int32_t wwama_tensor_write(
    llama_model * model,
    const char * name,
    size_t offset,
    const void * source,
    size_t size) {
    if (model == nullptr || name == nullptr || (source == nullptr && size != 0)) {
        return WWAMA_TENSOR_INVALID_ARGUMENT;
    }
    ggml_tensor * tensor = find_tensor(model, name);
    if (tensor == nullptr) {
        return WWAMA_TENSOR_NOT_FOUND;
    }
    if (!transfer_in_bounds(tensor, offset, size)) {
        return tensor->buffer == nullptr ? WWAMA_TENSOR_UNAVAILABLE : WWAMA_TENSOR_OUT_OF_BOUNDS;
    }
    if (size != 0) {
        ggml_backend_tensor_set(tensor, source, offset, size);
    }
    return WWAMA_TENSOR_OK;
}
