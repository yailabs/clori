/* Graph plans describe semantic execution and its checked resource geometry. They are immutable
 * after admission and remain distinct from backend launch graphs and session-owned persistent
 * state. */
#ifndef YVEX_GRAPH_H
#define YVEX_GRAPH_H

#include <yvex/model.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Graph descriptors. */
typedef struct yvex_graph yvex_graph;

typedef enum {
    YVEX_GRAPH_STATUS_EMPTY = 0,
    YVEX_GRAPH_STATUS_BUILT,
    YVEX_GRAPH_STATUS_PARTIAL,
    YVEX_GRAPH_STATUS_UNSUPPORTED,
    YVEX_GRAPH_STATUS_INVALID
} yvex_graph_status;

typedef struct {
    unsigned long long sequence_length;
    unsigned long long context_length;
    int include_decode_step;
    int include_prefill_path;
} yvex_graph_build_options;

typedef struct {
    yvex_tensor_role role;
    const char *role_name;
    const char *reason;
} yvex_graph_missing_required;

int yvex_graph_build_for_model(yvex_graph **out,
                                const yvex_model_descriptor *model,
                                const yvex_tensor_table *tensors,
                                const yvex_graph_build_options *options,
                                yvex_error *err);

void yvex_graph_close(yvex_graph *graph);

yvex_graph_status yvex_graph_status_of(const yvex_graph *graph);

unsigned long long yvex_graph_value_count(const yvex_graph *graph);
unsigned long long yvex_graph_op_count(const yvex_graph *graph);
unsigned long long yvex_graph_missing_required_count(const yvex_graph *graph);

const yvex_graph_value_info *yvex_graph_value_at(const yvex_graph *graph,
                                                 unsigned long long index);
const yvex_graph_op_info *yvex_graph_op_at(const yvex_graph *graph,
                                           unsigned long long index);
const yvex_graph_missing_required *yvex_graph_missing_required_at(const yvex_graph *graph,
                                                                  unsigned long long index);

int yvex_shape_product(const unsigned long long *dims,
                       unsigned int rank,
                       unsigned long long *out,
                       yvex_error *err);
int yvex_shape_equal(const unsigned long long *a,
                     unsigned int a_rank,
                     const unsigned long long *b,
                     unsigned int b_rank);
int yvex_shape_copy(unsigned long long *dst,
                    unsigned int dst_cap,
                    const unsigned long long *src,
                    unsigned int src_rank,
                    yvex_error *err);

/* Memory planning. */
typedef struct yvex_memory_plan yvex_memory_plan;

typedef enum {
    YVEX_MEMORY_PLAN_EMPTY = 0,
    YVEX_MEMORY_PLAN_ESTIMATED,
    YVEX_MEMORY_PLAN_PARTIAL,
    YVEX_MEMORY_PLAN_UNSUPPORTED
} yvex_memory_plan_status;

typedef struct {
    unsigned long long model_tensor_bytes_known;
    unsigned long long model_tensor_bytes_unknown_count;
    unsigned long long activation_peak_bytes;
    unsigned long long kv_cache_bytes;
    unsigned long long scratch_peak_bytes;
    unsigned long long total_known_bytes;
} yvex_memory_plan_summary;

int yvex_memory_plan_from_graph(yvex_memory_plan **out,
                                const yvex_graph *graph,
                                const yvex_tensor_table *tensors,
                                yvex_error *err);

void yvex_memory_plan_close(yvex_memory_plan *plan);

yvex_memory_plan_status yvex_memory_plan_status_of(const yvex_memory_plan *plan);

int yvex_memory_plan_get_summary(const yvex_memory_plan *plan,
                                 yvex_memory_plan_summary *out,
                                 yvex_error *err);

/* Execution planning. */
typedef struct yvex_plan yvex_plan;

typedef struct {
    unsigned long long sequence_length;
    unsigned long long context_length;
    const char *backend_name;
} yvex_plan_options;

int yvex_plan_create(yvex_plan **out,
                     const yvex_model_descriptor *model,
                     const yvex_tensor_table *tensors,
                     const yvex_plan_options *options,
                     yvex_error *err);

void yvex_plan_close(yvex_plan *plan);

const yvex_graph *yvex_plan_graph(const yvex_plan *plan);
const yvex_memory_plan *yvex_plan_memory(const yvex_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* YVEX_GRAPH_H */
