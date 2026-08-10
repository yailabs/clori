/* Bounded host buffers and exact resident weight views for admitted component execution. */
#ifndef INCLUDE_YVEX_INTERNAL_COMPONENT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_COMPONENT_H_INCLUDED

#include <yvex/internal/artifact.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yvex_runtime_residency yvex_runtime_residency;

typedef struct {
    float *data;
    unsigned long long count;
} yvex_component_f32_buffer;

typedef struct yvex_component_encoded_weight {
    const unsigned char *encoded;
    unsigned long long encoded_bytes, row_count, row_width, row_bytes;
    unsigned int qtype;
} yvex_component_encoded_weight;

typedef enum {
    YVEX_COMPONENT_LOAD_NONE = 0,
    YVEX_COMPONENT_LOAD_MISSING,
    YVEX_COMPONENT_LOAD_CONTRACT,
    YVEX_COMPONENT_LOAD_BUDGET,
    YVEX_COMPONENT_LOAD_MATERIALIZATION
} yvex_component_load_code;

typedef struct {
    yvex_component_load_code code;
    char tensor_name[256];
    unsigned long long expected, actual;
    const char *reason;
} yvex_component_load_failure;

int yvex_component_buffer_open(
    yvex_component_f32_buffer *, unsigned long long, unsigned long long,
    unsigned long long *, unsigned long long *, const char *, const char *, yvex_error *);
void yvex_component_buffer_close(yvex_component_f32_buffer *, unsigned long long *);
const yvex_materialized_tensor_binding *yvex_component_binding_find(
    const yvex_materialization_session *, const char *);
int yvex_component_weight_bind(
    const yvex_materialization_session *, const yvex_runtime_residency *,
    const char *, yvex_component_encoded_weight *, yvex_error *);
int yvex_component_weight_bind_sized(
    void *, const char *, unsigned long long, unsigned long long,
    yvex_component_encoded_weight *, yvex_error *);
int yvex_component_f32_load(
    yvex_materialization_session *, const char *, unsigned int,
    const unsigned long long *, yvex_component_f32_buffer *, unsigned long long,
    unsigned long long *, unsigned long long *, unsigned long long *,
    unsigned long long *, yvex_component_load_failure *, const char *, const char *,
    yvex_error *);

#ifdef __cplusplus
}
#endif
#endif
