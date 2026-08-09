/*
 * A launch-graph object owns or borrows one stream and admits at most one CUDA graph/executable
 * pair. Capture is exclusive within a backend because Driver capture state and the temporary
 * operation inventory must describe one coherent launch sequence.
 *
 * Updates preserve the prior executable graph until compatibility and Driver admission succeed.
 * Graph replay only schedules operations already admitted by backend owners; it never infers model
 * topology or supplies a numerical fallback.
 */
#include "src/backend/cuda/private.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CUDA_GRAPH_UPDATE_SUCCESS 0
typedef struct {
    unsigned long long from;
    unsigned long long to;
    CUgraphEdgeData data;
} cuda_edge_fact;
typedef struct {
    char identity[YVEX_SHA256_HEX_BYTES];
    CUfunction function;
    unsigned int grid, block, shared_bytes;
} cuda_kernel_signature;
typedef struct {
    CUgraphNode node;
    char identity[YVEX_SHA256_HEX_BYTES];
} cuda_kernel_binding;
struct yvex_backend_cuda_graph {
    yvex_backend *backend;
    yvex_backend_cuda_graph *next;
    yvex_backend_cuda_graph_state state;
    yvex_backend_cuda_graph_state capture_origin_state;
    yvex_backend_cuda_graph_reason reason;
    yvex_backend_cuda_capture_mode capture_mode;
    char *compatibility_identity;
    CUstream stream;
    int stream_owned;
    CUgraph graph;
    CUgraph pending_graph;
    CUgraphExec exec;
    cuda_kernel_binding *kernel_bindings;
    size_t kernel_node_count;
    size_t kernel_update_cursor;
    cuda_kernel_signature *capture_kernels;
    size_t capture_kernel_count;
    size_t capture_kernel_capacity;
    int uploaded;
    yvex_backend_cuda_graph_inventory inventory;
    unsigned long long capture_count;
    unsigned long long instantiate_count;
    unsigned long long upload_count;
    unsigned long long update_count;
    unsigned long long launch_count;
    unsigned long long replay_count;
    unsigned long long synchronize_count;
    unsigned long long invalidation_count;
    unsigned long long workspace_cursor;
    unsigned long long capture_started_ns;
    unsigned long long capture_elapsed_ns;
    unsigned long long instantiate_elapsed_ns;
    unsigned long long last_update_elapsed_ns;
    unsigned long long last_replay_elapsed_ns;
    unsigned long long last_device_elapsed_ns;
    int update_requested;
    int in_flight;
    char launch_graph_identity[YVEX_BACKEND_CUDA_GRAPH_IDENTITY_CAP];
    char graph_exec_identity[YVEX_BACKEND_CUDA_GRAPH_IDENTITY_CAP];
};
static int graph_info(const yvex_backend_cuda_graph *graph,
                      yvex_backend_cuda_graph_info *out, yvex_error *err);
static int graph_release(yvex_backend_cuda_graph **graph, yvex_error *err);
static const char *const graph_reason_names[] = {
    "none", "not-cuda", "context-unavailable", "backend-failed",
    "stream-api-unavailable", "graph-api-unavailable", "update-api-unavailable", "busy",
    "invalid-state", "capture-failed", "empty-capture", "instantiate-failed",
    "upload-failed", "update-incompatible", "launch-failed", "synchronize-failed", "cleanup-failed"
};

static int graph_failure_matches(const char *stage)
{
    const char *selected = getenv("YVEX_TEST_CUDA_GRAPH_FAILURE");
    return selected && stage && (strcmp(selected, "all") == 0 || strcmp(selected, stage) == 0);
}

static void graph_capture_restore(yvex_backend_cuda_graph *graph,
                                  yvex_backend_cuda_graph_reason reason)
{
    graph->state = graph->capture_origin_state;
    graph->reason = reason;
}

static void graph_poison(yvex_backend_cuda_graph *graph)
{
    graph->state = YVEX_BACKEND_CUDA_GRAPH_INVALIDATED;
    graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NONE;
}

static void graph_mark_failed(yvex_backend_cuda_graph *graph,
                              yvex_backend_cuda_graph_reason reason)
{
    graph->state = YVEX_BACKEND_CUDA_GRAPH_FAILED;
    graph->reason = reason;
}

static int graph_reject(yvex_error *err, int code, const char *where, const char *message)
{
    yvex_error_set(err, code, where, message);
    return code;
}

static int graph_fail(yvex_backend_cuda_graph *graph, yvex_backend_cuda_graph_state state,
                      yvex_backend_cuda_graph_reason reason, int code, const char *where,
                      const char *message, yvex_error *err)
{
    graph->state = state;
    graph->reason = reason;
    yvex_error_set(err, code, where, message);
    return code;
}

static int graph_cleanup_result(yvex_backend_cuda_graph *graph, int rc)
{
    if (rc != YVEX_OK) {
        graph->state = YVEX_BACKEND_CUDA_GRAPH_FAILED;
        graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_CLEANUP_FAILED;
    }
    return rc;
}

static int graph_cleanup_fail(yvex_backend_cuda_graph *graph, const char *where,
                              const char *message, yvex_error *err)
{
    return graph_fail(graph, YVEX_BACKEND_CUDA_GRAPH_FAILED,
                      YVEX_BACKEND_CUDA_GRAPH_REASON_CLEANUP_FAILED,
                      YVEX_ERR_BACKEND, where, message, err);
}

static int stream_api_available(const yvex_cuda_driver *driver)
{
    return driver && driver->cuStreamCreate && driver->cuStreamDestroy_v2 &&
           driver->cuStreamSynchronize && driver->cuStreamBeginCapture_v2 &&
           driver->cuStreamEndCapture;
}

static int graph_api_available(const yvex_cuda_driver *driver)
{
    return driver && driver->cuGraphGetNodes && driver->cuGraphNodeGetType &&
           driver->cuGraphGetEdges_v2 &&
           driver->cuGraphExecKernelNodeSetParams_v2 &&
           driver->cuGraphInstantiateWithFlags && driver->cuGraphUpload &&
           driver->cuGraphLaunch && driver->cuGraphExecDestroy && driver->cuGraphDestroy;
}

static int edge_fact_compare(const void *left, const void *right)
{
    const cuda_edge_fact *a = (const cuda_edge_fact *)left;
    const cuda_edge_fact *b = (const cuda_edge_fact *)right;
    if (a->from != b->from) return a->from < b->from ? -1 : 1;
    if (a->to != b->to) return a->to < b->to ? -1 : 1;
    if (a->data.from_port != b->data.from_port)
        return a->data.from_port < b->data.from_port ? -1 : 1;
    if (a->data.to_port != b->data.to_port)
        return a->data.to_port < b->data.to_port ? -1 : 1;
    if (a->data.type != b->data.type)
        return a->data.type < b->data.type ? -1 : 1;
    return 0;
}

static int node_index(const CUgraphNode *nodes, size_t count, CUgraphNode node,
                      unsigned long long *out)
{
    size_t index;
    for (index = 0u; index < count; ++index) {
        if (nodes[index] == node) {
            *out = (unsigned long long)index;
            return 1;
        }
    }
    return 0;
}

static void inventory_node(yvex_backend_cuda_graph_inventory *inventory, int type)
{
    switch (type) {
    case 0: inventory->kernel_node_count++; break;
    case 1: inventory->memcpy_node_count++; break;
    case 2: inventory->memset_node_count++; break;
    case 3: inventory->host_node_count++; break;
    case 4: inventory->child_graph_node_count++; break;
    case 6: case 7: inventory->event_node_count++; break;
    case 10: case 11: inventory->memory_node_count++; break;
    default: inventory->other_node_count++; break;
    }
}

static int identity_finish(yvex_sha256 *sha, char output[65], yvex_error *err)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(sha, digest)) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.identity",
                            "CUDA graph identity stream could not be finalized");
    }
    yvex_sha256_hex(digest, output);
    return YVEX_OK;
}

static const char *kernel_function_identity(const yvex_cuda_backend_state *state,
                                            CUfunction function)
{
#define MATCH(MEMBER) if (function == state->MEMBER) return #MEMBER
    MATCH(embed_function); MATCH(embed_f16_function); MATCH(rms_norm_f32_function);
    MATCH(rms_norm_f16_function); MATCH(rope_function); MATCH(matmul_function);
    MATCH(qtype_row_dot_function); MATCH(attention_bf16_round_function);
    MATCH(qtype_matvec_function); MATCH(qtype_split_matvec_function);
    MATCH(qtype_gather_function); MATCH(q8_quantize_function);
    MATCH(encoded_row_decode_function); MATCH(attention_weighted_norm_function);
    MATCH(attention_unit_norm_function); MATCH(attention_yarn_rope_function);
    MATCH(attention_activation_quantize_function); MATCH(residual_mhc_pre_function);
    MATCH(residual_mhc_post_function); MATCH(transformer_feature_mean_function);
    MATCH(transformer_final_function);
    MATCH(attention_rolling_state_function);
    MATCH(attention_topk_function); MATCH(attention_reduce_function);
    MATCH(argmax_f32_function); MATCH(sample_stochastic_f32_function);
    MATCH(speculation_stochastic_f32_function);
    MATCH(moe_route_function); MATCH(moe_route_rows_function); MATCH(moe_pair_order_function);
    MATCH(moe_grouped_up_function); MATCH(moe_grouped_down_function);
    MATCH(moe_grouped_up_rows_function); MATCH(moe_grouped_down_rows_function);
    MATCH(moe_reduce_rows_function); MATCH(moe_combine_rows_function);
    MATCH(moe_swiglu_function); MATCH(moe_accumulate_function);
    MATCH(mlp_function); MATCH(attention_function);
#undef MATCH
    return NULL;
}

static int kernel_signature(const yvex_backend *backend, yvex_backend_operation_variant variant,
                            CUfunction function, unsigned int grid, unsigned int block,
                            unsigned int shared_bytes,
                            const char *stage, char output[65], yvex_error *err)
{
    const yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    const char *function_identity = state ? kernel_function_identity(state, function) : NULL;
    yvex_sha256 sha;
    if (!state || !function_identity || !stage || !stage[0] || !grid || !block ||
        !yvex_sha256_hex_valid(state->kernel_bundle_identity)) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.kernel_signature",
                            "admitted bundle, function, stage, and launch geometry are required");
    }
    yvex_sha256_init(&sha);
    if (!yvex_sha256_update_text(&sha, "yvex.cuda.kernel-launch.v1") ||
        !yvex_sha256_update_text(&sha, state->kernel_bundle_identity) ||
        !yvex_sha256_update_text(&sha, function_identity) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)variant) ||
        !yvex_sha256_update_text(&sha, stage) ||
        !yvex_sha256_update_u64(&sha, grid) || !yvex_sha256_update_u64(&sha, block) ||
        !yvex_sha256_update_u64(&sha, shared_bytes)) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.kernel_signature",
                            "CUDA kernel signature serialization failed");
    }
    return identity_finish(&sha, output, err);
}
/*
 * Retain one successful captured launch signature for deterministic node admission.
 *
 * Active capture owner and the exact admitted launch compatibility facts. Inactive capture,
 * overflow, allocation, or identity failure appends nothing. Capture registration allocates;
 * instantiated graph replay never does.
 */
int yvex_cuda_graph_kernel_capture(yvex_backend *backend,
                                   yvex_backend_operation_variant variant, CUfunction function,
                                   unsigned int grid, unsigned int block, unsigned int shared_bytes,
                                   const char *stage, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_backend_cuda_graph *graph = state ? state->capture_owner : NULL;
    cuda_kernel_signature *grown;
    size_t capacity;
    int rc;
    if (!graph || graph->state != YVEX_BACKEND_CUDA_GRAPH_CAPTURING) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.kernel_capture",
                            "an active CUDA graph capture is required");
    }
    if (graph->capture_kernel_count == graph->capture_kernel_capacity) {
        capacity = graph->capture_kernel_capacity ? graph->capture_kernel_capacity * 2u : 32u;
        if (capacity < graph->capture_kernel_capacity || capacity > SIZE_MAX / sizeof(*grown)) {
            return graph_reject(err, YVEX_ERR_BOUNDS, "cuda.graph.kernel_capture",
                                "captured kernel signature count overflowed");
        }
        grown = (cuda_kernel_signature *)realloc(graph->capture_kernels, capacity * sizeof(*grown));
        if (!grown) {
            return graph_reject(err, YVEX_ERR_NOMEM, "cuda.graph.kernel_capture",
                                "captured kernel signature allocation failed");
        }
        graph->capture_kernels = grown;
        graph->capture_kernel_capacity = capacity;
    }
    rc = kernel_signature(backend, variant, function, grid, block, shared_bytes, stage,
                          graph->capture_kernels[graph->capture_kernel_count].identity, err);
    if (rc == YVEX_OK) {
        cuda_kernel_signature *signature =
            &graph->capture_kernels[graph->capture_kernel_count++];
        signature->function = function;
        signature->grid = grid;
        signature->block = block;
        signature->shared_bytes = shared_bytes;
    }
    return rc;
}

static int exec_identity(const yvex_backend_cuda_graph *owner, const char *graph_identity,
                         char output[65], yvex_error *err)
{
    const yvex_cuda_backend_state *state = yvex_cuda_state(owner->backend);
    yvex_sha256 sha;
    yvex_sha256_init(&sha);
    if (!yvex_sha256_update_text(&sha, "yvex.cuda.graph-exec.v2") ||
        !yvex_sha256_update_text(&sha, graph_identity) ||
        !yvex_sha256_update_text(&sha, state->kernel_bundle_identity) ||
        !yvex_sha256_update_u64(&sha, YVEX_BACKEND_CUDA_GRAPH_SCHEMA) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)state->driver_version) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)state->device_index) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)!owner->stream_owned)) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.identity",
                            "CUDA graph executable identity stream rejected input");
    }
    return identity_finish(&sha, output, err);
}
/*
 * Bind canonical node order and captured kernel signatures into one launch identity.
 *
 * Digest serialization failure publishes no usable identity. Driver handles are retained for
 * replay but never hashed.
 */
static int inventory_identity(const yvex_backend_cuda_graph *owner, const int *types,
                              const size_t *order, size_t node_count,
                              const cuda_edge_fact *edges, size_t edge_count,
                              char (*node_identities)[YVEX_SHA256_HEX_BYTES],
                              char identity[65], yvex_error *err)
{
    const yvex_cuda_backend_state *state = yvex_cuda_state(owner->backend);
    yvex_sha256 sha;
    size_t index, position;
    yvex_sha256_init(&sha);
    if (!yvex_sha256_update_text(&sha, "yvex.cuda.launch-graph.v3") ||
        !yvex_sha256_update_text(&sha, owner->compatibility_identity) ||
        !yvex_sha256_update_text(&sha, state->kernel_bundle_identity) ||
        !yvex_sha256_update_u64(&sha, YVEX_BACKEND_CUDA_GRAPH_SCHEMA) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)state->driver_version) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)state->device_index) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)owner->capture_mode) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)!owner->stream_owned) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)node_count) ||
        !yvex_sha256_update_u64(&sha, (unsigned long long)edge_count) ||
        !yvex_sha256_update_u64(&sha, 1ull))
        goto failed;
    for (position = 0u; position < node_count; ++position) {
        index = order[position];
        if (!yvex_sha256_update_u64(&sha, (unsigned long long)types[index]))
            goto failed;
        if (types[index] == 0 &&
            !yvex_sha256_update_text(&sha, node_identities[index]))
            goto failed;
    }
    for (index = 0u; index < edge_count; ++index) {
        if (!yvex_sha256_update_u64(&sha, edges[index].from) ||
            !yvex_sha256_update_u64(&sha, edges[index].to) ||
            !yvex_sha256_update_u64(&sha, edges[index].data.from_port) ||
            !yvex_sha256_update_u64(&sha, edges[index].data.to_port) ||
            !yvex_sha256_update_u64(&sha, edges[index].data.type))
            goto failed;
    }
    return identity_finish(&sha, identity, err);
failed:
    return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.identity",
                        "CUDA launch graph identity stream rejected topology facts");
}
/*
 * Admit one captured graph through canonical dependency order and registered kernel work.
 *
 * Captured graph plus successful pointer-free kernel signatures recorded during capture. Returns
 * inventory, identity, and exact kernel-node bindings for allocation-free replay.
 */
static int graph_inventory(yvex_backend_cuda_graph *owner, CUgraph candidate,
                           yvex_backend_cuda_graph_inventory *inventory,
                           cuda_kernel_binding **out_bindings, size_t *out_binding_count,
                           char identity[65], yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(owner->backend);
    CUgraphNode *nodes = NULL;
    int *types = NULL;
    cuda_edge_fact *edges = NULL;
    CUgraphNode *from = NULL, *to = NULL;
    CUgraphEdgeData *edge_data = NULL;
    size_t *indegree = NULL, *order = NULL, *canonical = NULL;
    char (*node_identities)[YVEX_SHA256_HEX_BYTES] = NULL;
    cuda_kernel_binding *bindings = NULL;
    size_t node_count = 0u, edge_count = 0u;
    size_t index, edge, position, selected, roots, kernel = 0u;
    int rc = YVEX_OK;
    memset(inventory, 0, sizeof(*inventory));
    *out_bindings = NULL;
    *out_binding_count = 0u;
    identity[0] = '\0';
    if (graph_failure_matches("inventory")) {
        return graph_reject(err, YVEX_ERR_BACKEND, "cuda.graph.inventory",
                            "injected CUDA graph inventory failure");
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuGraphGetNodes(candidate, NULL, &node_count),
                          "cuda.graph.nodes", err);
    if (rc != YVEX_OK) return rc;
    if (node_count == 0u) {
        return graph_reject(err, YVEX_ERR_UNSUPPORTED, "cuda.graph.inventory",
                            "CUDA graph capture produced no executable nodes");
    }
    if (node_count > SIZE_MAX / sizeof(*nodes) || node_count > SIZE_MAX / sizeof(*types) ||
        node_count > SIZE_MAX / sizeof(*indegree)) {
        return graph_reject(err, YVEX_ERR_BOUNDS, "cuda.graph.inventory",
                            "CUDA graph node inventory exceeds host bounds");
    }
    nodes = (CUgraphNode *)calloc(node_count, sizeof(*nodes));
    types = (int *)calloc(node_count, sizeof(*types));
    indegree = (size_t *)calloc(node_count, sizeof(*indegree));
    order = (size_t *)calloc(node_count, sizeof(*order));
    canonical = (size_t *)calloc(node_count, sizeof(*canonical));
    node_identities = calloc(node_count, sizeof(*node_identities));
    if (!nodes || !types || !indegree || !order || !canonical || !node_identities) {
        rc = graph_reject(err, YVEX_ERR_NOMEM, "cuda.graph.inventory",
                          "failed to allocate CUDA graph node inventory");
        goto done;
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuGraphGetNodes(candidate, nodes, &node_count),
                          "cuda.graph.nodes", err);
    if (rc != YVEX_OK) goto done;
    if (getenv("YVEX_TEST_CUDA_GRAPH_NODE_ORDER")) {
        for (index = 0u; index < node_count / 2u; ++index) {
            CUgraphNode swap = nodes[index];
            nodes[index] = nodes[node_count - index - 1u];
            nodes[node_count - index - 1u] = swap;
        }
    }
    inventory->node_count = (unsigned long long)node_count;
    for (index = 0u; index < node_count; ++index) {
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuGraphNodeGetType(nodes[index], &types[index]),
                              "cuda.graph.node_type", err);
        if (rc != YVEX_OK) goto done;
        inventory_node(inventory, types[index]);
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuGraphGetEdges_v2(candidate, NULL, NULL, NULL, &edge_count),
                          "cuda.graph.edges", err);
    if (rc != YVEX_OK) goto done;
    if (edge_count > 0u) {
        if (edge_count > SIZE_MAX / sizeof(*from) ||
            edge_count > SIZE_MAX / sizeof(*edge_data) || edge_count > SIZE_MAX / sizeof(*edges)) {
            rc = graph_reject(err, YVEX_ERR_BOUNDS, "cuda.graph.inventory",
                              "CUDA graph edge inventory exceeds host bounds");
            goto done;
        }
        from = (CUgraphNode *)calloc(edge_count, sizeof(*from));
        to = (CUgraphNode *)calloc(edge_count, sizeof(*to));
        edge_data = (CUgraphEdgeData *)calloc(edge_count, sizeof(*edge_data));
        edges = (cuda_edge_fact *)calloc(edge_count, sizeof(*edges));
        if (!from || !to || !edge_data || !edges) {
            rc = graph_reject(err, YVEX_ERR_NOMEM, "cuda.graph.inventory",
                              "failed to allocate CUDA graph edge inventory");
            goto done;
        }
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuGraphGetEdges_v2(candidate, from, to, edge_data,
                                                               &edge_count),
                              "cuda.graph.edges", err);
        if (rc != YVEX_OK) goto done;
        for (index = 0u; index < edge_count; ++index) {
            if (!node_index(nodes, node_count, from[index], &edges[index].from) ||
                !node_index(nodes, node_count, to[index], &edges[index].to)) {
                rc = graph_reject(err, YVEX_ERR_FORMAT, "cuda.graph.inventory",
                                  "CUDA graph edge references an unknown node");
                goto done;
            }
            edges[index].data = edge_data[index];
            if (indegree[edges[index].to] == SIZE_MAX) {
                rc = graph_reject(err, YVEX_ERR_BOUNDS, "cuda.graph.inventory",
                                  "CUDA graph dependency count overflowed");
                goto done;
            }
            indegree[edges[index].to]++;
        }
    }
    for (position = 0u; position < node_count; ++position) {
        selected = SIZE_MAX;
        roots = 0u;
        for (index = 0u; index < node_count; ++index) {
            if (indegree[index] == 0u) {
                selected = index;
                roots++;
            }
        }
        if (roots != 1u) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "cuda.graph.inventory",
                            "CUDA graph dependency order is ambiguous; roots=%zu", roots);
            rc = YVEX_ERR_FORMAT;
            goto done;
        }
        indegree[selected] = SIZE_MAX;
        order[position] = selected;
        canonical[selected] = position;
        for (edge = 0u; edge < edge_count; ++edge) {
            if (edges[edge].from != selected) continue;
            if (!indegree[edges[edge].to]) {
                rc = graph_reject(err, YVEX_ERR_FORMAT, "cuda.graph.inventory",
                                  "CUDA graph dependency accounting is inconsistent");
                goto done;
            }
            indegree[edges[edge].to]--;
        }
    }
    if (!inventory->kernel_node_count ||
        inventory->kernel_node_count != owner->capture_kernel_count) {
        yvex_error_setf(err, YVEX_ERR_FORMAT, "cuda.graph.kernel_nodes",
                        "captured kernel registration mismatch; nodes=%llu signatures=%zu",
                        inventory->kernel_node_count, owner->capture_kernel_count);
        rc = YVEX_ERR_FORMAT;
        goto done;
    }
    bindings = (cuda_kernel_binding *)calloc(owner->capture_kernel_count, sizeof(*bindings));
    if (!bindings) {
        rc = graph_reject(err, YVEX_ERR_NOMEM, "cuda.graph.kernel_nodes",
                          "captured kernel binding allocation failed");
        goto done;
    }
    for (position = 0u; position < node_count; ++position) {
        index = order[position];
        if (types[index] != 0) continue;
        bindings[kernel].node = nodes[index];
        memcpy(bindings[kernel].identity, owner->capture_kernels[kernel].identity,
               sizeof(bindings[kernel].identity));
        memcpy(node_identities[index], bindings[kernel].identity,
               sizeof(node_identities[index]));
        kernel++;
    }
    for (edge = 0u; edge < edge_count; ++edge) {
        edges[edge].from = canonical[edges[edge].from];
        edges[edge].to = canonical[edges[edge].to];
    }
    qsort(edges, edge_count, sizeof(*edges), edge_fact_compare);
    inventory->edge_count = (unsigned long long)edge_count;
    rc = inventory_identity(owner, types, order, node_count, edges, edge_count,
                            node_identities, identity, err);
    if (rc == YVEX_OK) {
        *out_bindings = bindings;
        *out_binding_count = owner->capture_kernel_count;
        bindings = NULL;
    }
done:
    free(node_identities);
    free(bindings);
    free(canonical);
    free(order);
    free(indegree);
    free(edges);
    free(edge_data);
    free(to);
    free(from);
    free(types);
    free(nodes);
    return rc;
}

static int candidate_destroy(yvex_backend_cuda_graph *owner, CUgraph candidate, int primary_rc,
                             yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(owner->backend);
    yvex_error cleanup_error;
    int rc;
    if (!candidate) return primary_rc;
    if (graph_failure_matches("graph-destroy")) {
        if (candidate != owner->graph && !owner->pending_graph)
            owner->pending_graph = candidate;
        return graph_cleanup_fail(owner, "cuda.graph.destroy",
                                  "injected CUDA graph cleanup failure", err);
    }
    yvex_error_clear(&cleanup_error);
    rc = yvex_cuda_status(&state->driver, state->driver.cuGraphDestroy(candidate),
                          "cuda.graph.destroy", &cleanup_error);
    if (rc != YVEX_OK) {
        if (candidate != owner->graph && !owner->pending_graph)
            owner->pending_graph = candidate;
        graph_cleanup_result(owner, rc);
        if (err) *err = cleanup_error;
    }
    return rc == YVEX_OK ? primary_rc : rc;
}

static void graph_unlink(yvex_backend_cuda_graph *graph)
{
    yvex_cuda_backend_state *state;
    yvex_backend_cuda_graph **cursor;
    if (!graph || !graph->backend) return;
    state = yvex_cuda_state(graph->backend);
    if (!state) return;
    cursor = &state->graphs;
    while (*cursor) {
        if (*cursor == graph) {
            *cursor = graph->next;
            graph->next = NULL;
            return;
        }
        cursor = &(*cursor)->next;
    }
}

static yvex_backend_cuda_graph *graph_find(yvex_cuda_backend_state *state, const char *identity,
                                           int shared_launch_stream)
{
    yvex_backend_cuda_graph *graph;
    if (!state || !identity) return NULL;
    for (graph = state->graphs; graph; graph = graph->next) {
        if (graph->compatibility_identity &&
            strcmp(graph->compatibility_identity, identity) == 0 &&
            shared_launch_stream == !graph->stream_owned)
            return graph;
    }
    return NULL;
}

static int graph_is_attention(const yvex_cuda_backend_state *state,
                              const yvex_backend_cuda_graph *graph)
{
    unsigned int index;
    size_t graph_length, length;
    if (!state || !graph || !graph->compatibility_identity)
        return 0;
    graph_length = strlen(graph->compatibility_identity);
    for (index = 0u; index < state->attention_configuration_count; ++index) {
        const yvex_cuda_attention_configuration *configuration =
            &state->attention_configurations[index];
        length = strlen(configuration->compatibility_identity);
        if (configuration->configured && length < graph_length &&
            strncmp(graph->compatibility_identity,
                    configuration->compatibility_identity, length) == 0 &&
            graph->compatibility_identity[length] == ':')
            return 1;
    }
    return 0;
}

const yvex_cuda_attention_configuration *yvex_cuda_attention_configuration_active(
    const yvex_cuda_backend_state *state, yvex_backend_attention_phase phase) {
    const yvex_cuda_attention_configuration *configuration;
    if (!state || !state->attention_configuration_count ||
        state->attention_active_configuration >=
            state->attention_configuration_count)
        return NULL;
    configuration =
        &state->attention_configurations[state->attention_active_configuration];
    return configuration->configured && configuration->phase == phase
               ? configuration
               : NULL;
}
/*
 * Return the stream selected by an explicit graph lifecycle or the session execution owner.
 *
 * Graph capture and parameter refresh must stay on the graph stream. Ordinary kernels use the
 * independently owned session stream so their completion need not stall the complete context.
 */
CUstream yvex_cuda_launch_stream(const yvex_backend *backend)
{
    const yvex_cuda_backend_state *state =
        backend && backend->kind == YVEX_BACKEND_KIND_CUDA ? yvex_cuda_state(backend) : NULL;
    if (!state) return NULL;
    if (state->capture_owner) return state->capture_stream;
    return state->parameter_update_owner ? state->parameter_update_owner->stream
                                         : state->execution_stream;
}
/*
 * Report whether a backend is inside one explicit, exclusive stream capture.
 *
 * Missing state reports inactive capture.
 */
int yvex_cuda_capture_active(const yvex_backend *backend)
{
    const yvex_cuda_backend_state *state =
        backend && backend->kind == YVEX_BACKEND_KIND_CUDA ? yvex_cuda_state(backend) : NULL;
    return state && state->capture_owner && state->capture_stream;
}

static const char *graph_reason_name(yvex_backend_cuda_graph_reason reason)
{
    size_t count = sizeof(graph_reason_names) / sizeof(graph_reason_names[0]);
    return reason >= 0 && (size_t)reason < count ? graph_reason_names[reason] : "unknown";
}

int yvex_backend_cuda_graph_query(const yvex_backend *backend,
                                  yvex_backend_cuda_graph_capability *out, yvex_error *err)
{
    const yvex_cuda_backend_state *state;
    if (!backend || !out) {
        return graph_reject(err, YVEX_ERR_INVALID_ARG, "cuda.graph.query",
                            "backend and capability output are required");
    }
    memset(out, 0, sizeof(*out));
    out->schema = YVEX_BACKEND_CUDA_GRAPH_SCHEMA;
    out->state = YVEX_BACKEND_CUDA_GRAPH_UNAVAILABLE;
    if (backend->kind != YVEX_BACKEND_KIND_CUDA) {
        out->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NOT_CUDA;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    state = yvex_cuda_state(backend);
    if (!state || !state->context) {
        out->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_CONTEXT_UNAVAILABLE;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (backend_cleanup_only(backend) ||
        backend->status == YVEX_BACKEND_STATUS_FAILED) {
        out->state = YVEX_BACKEND_CUDA_GRAPH_FAILED;
        out->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_BACKEND_FAILED;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    out->stream_api_available = stream_api_available(&state->driver);
    out->graph_api_available = graph_api_available(&state->driver);
    out->update_api_available = state->driver.cuGraphExecUpdate_v2 != NULL;
    out->edge_inventory_available = state->driver.cuGraphGetEdges_v2 != NULL;
    out->async_memory_available = state->driver.cuMemAllocAsync && state->driver.cuMemFreeAsync;
    out->async_copy_available = state->driver.cuMemcpyHtoDAsync_v2 &&
                                state->driver.cuMemcpyDtoHAsync_v2 &&
                                state->driver.cuMemsetD8Async;
    out->pinned_host_memory_available = state->driver.cuMemHostAlloc && state->driver.cuMemFreeHost;
    out->event_timing_available = state->driver.cuEventCreate && state->driver.cuEventRecord &&
                                  state->driver.cuEventSynchronize &&
                                  state->driver.cuEventElapsedTime_v2 &&
                                  state->driver.cuEventDestroy_v2;
    if (getenv("YVEX_TEST_CUDA_GRAPH_API")) {
        out->stream_api_available = 0;
        out->graph_api_available = 0;
        out->update_api_available = 0;
    }
    if (!out->stream_api_available) {
        out->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_STREAM_API_UNAVAILABLE;
    } else if (!out->graph_api_available) {
        out->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_GRAPH_API_UNAVAILABLE;
    } else if (!out->update_api_available) {
        out->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_UPDATE_API_UNAVAILABLE;
    } else {
        out->state = YVEX_BACKEND_CUDA_GRAPH_OPEN;
        out->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NONE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Create one stream-backed launch-graph object after complete optional API admission.
 *
 * A CUDA backend, schema/capture options, stream policy, and caller-owned result storage. The
 * graph either creates an isolated non-blocking stream or borrows the session execution stream;
 * borrowed ownership never escapes or destroys that session resource. API, schema, allocation,
 * or stream failures publish no partial graph object. Compatibility identity is caller-provided
 * execution truth, never inferred from a family name.
 */
static int graph_open(yvex_backend *backend,
                      const yvex_backend_cuda_graph_options *options,
                      int shared_launch_stream,
                      yvex_backend_cuda_graph **out, yvex_error *err)
{
    yvex_backend_cuda_graph_capability capability;
    yvex_backend_cuda_graph *graph;
    yvex_cuda_backend_state *state;
    int rc;
    if (!backend || !options || !out || !options->compatibility_identity ||
        (shared_launch_stream != 0 && shared_launch_stream != 1) ||
        options->compatibility_identity[0] == '\0') {
        return graph_reject(err, YVEX_ERR_INVALID_ARG, "cuda.graph.open",
                            "backend, options, compatibility identity, and out are required");
    }
    *out = NULL;
    if (options->schema != YVEX_BACKEND_CUDA_GRAPH_SCHEMA ||
        options->capture_mode < YVEX_BACKEND_CUDA_CAPTURE_GLOBAL ||
        options->capture_mode > YVEX_BACKEND_CUDA_CAPTURE_RELAXED) {
        return graph_reject(err, YVEX_ERR_UNSUPPORTED, "cuda.graph.open",
                            "unsupported CUDA graph schema or capture mode");
    }
    rc = yvex_backend_cuda_graph_query(backend, &capability, err);
    if (rc != YVEX_OK) return rc;
    if (capability.state != YVEX_BACKEND_CUDA_GRAPH_OPEN) {
        int failure = capability.state == YVEX_BACKEND_CUDA_GRAPH_FAILED
                          ? YVEX_ERR_BACKEND : YVEX_ERR_UNSUPPORTED;
        yvex_error_setf(err, failure, "cuda.graph.open", "CUDA graph unavailable: %s",
                        graph_reason_name(capability.reason));
        return failure;
    }
    graph = (yvex_backend_cuda_graph *)calloc(1, sizeof(*graph));
    if (!graph) {
        return graph_reject(err, YVEX_ERR_NOMEM, "cuda.graph.open",
                            "failed to allocate CUDA graph lifecycle object");
    }
    graph->compatibility_identity = yvex_core_strdup(options->compatibility_identity);
    if (!graph->compatibility_identity) {
        free(graph);
        return graph_reject(err, YVEX_ERR_NOMEM, "cuda.graph.open",
                            "failed to copy CUDA graph compatibility identity");
    }
    state = yvex_cuda_state(backend);
    rc = yvex_cuda_set_current(backend, "cuda.graph.open", err);
    if (rc == YVEX_OK && !shared_launch_stream && graph_failure_matches("stream-create")) {
        rc = graph_reject(err, YVEX_ERR_BACKEND, "cuda.graph.open",
                          "injected CUDA graph stream creation failure");
    }
    if (rc == YVEX_OK && shared_launch_stream) {
        graph->stream = yvex_cuda_launch_stream(backend);
        if (!graph->stream)
            rc = graph_reject(err, YVEX_ERR_STATE, "cuda.graph.open",
                              "session execution stream is unavailable");
    } else if (rc == YVEX_OK) {
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuStreamCreate(&graph->stream,
                                                           YVEX_CUDA_STREAM_NON_BLOCKING),
                              "cuda.graph.stream_create", err);
        if (rc == YVEX_OK) graph->stream_owned = 1;
    }
    if (rc != YVEX_OK) {
        free(graph->compatibility_identity);
        free(graph);
        return rc;
    }
    graph->backend = backend;
    graph->state = YVEX_BACKEND_CUDA_GRAPH_OPEN;
    graph->capture_mode = options->capture_mode;
    graph->next = state->graphs;
    state->graphs = graph;
    *out = graph;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Begin exclusive stream capture while preserving any previously admitted executable graph.
 *
 * Busy, state, injection, or Driver failures leave no active capture selection.
 */
static int graph_begin(yvex_backend_cuda_graph *graph, yvex_error *err)
{
    yvex_cuda_backend_state *state;
    int rc;
    if (!graph || !graph->backend) {
        return graph_reject(err, YVEX_ERR_INVALID_ARG, "cuda.graph.begin",
                            "attached CUDA graph object is required");
    }
    if (graph->state != YVEX_BACKEND_CUDA_GRAPH_OPEN &&
        graph->state != YVEX_BACKEND_CUDA_GRAPH_INVALIDATED &&
        graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED) {
        return graph_fail(graph, graph->state, YVEX_BACKEND_CUDA_GRAPH_REASON_INVALID_STATE,
                          YVEX_ERR_STATE, "cuda.graph.begin",
                          "CUDA graph is not in a capturable lifecycle state", err);
    }
    if (graph->pending_graph) {
        return graph_fail(graph, graph->state, YVEX_BACKEND_CUDA_GRAPH_REASON_CLEANUP_FAILED,
                          YVEX_ERR_STATE, "cuda.graph.begin",
                          "CUDA graph has pending cleanup and cannot capture", err);
    }
    if (graph->in_flight) {
        return graph_fail(graph, graph->state, YVEX_BACKEND_CUDA_GRAPH_REASON_BUSY,
                          YVEX_ERR_STATE, "cuda.graph.begin",
                          "CUDA graph stream must be synchronized before recapture", err);
    }
    state = yvex_cuda_state(graph->backend);
    if (state->capture_owner) {
        return graph_fail(graph, graph->state, YVEX_BACKEND_CUDA_GRAPH_REASON_BUSY,
                          YVEX_ERR_STATE, "cuda.graph.begin",
                          "another CUDA graph capture is active on this backend", err);
    }
    rc = yvex_cuda_set_current(graph->backend, "cuda.graph.begin", err);
    if (rc != YVEX_OK) return rc;
    if (graph_failure_matches("capture-begin")) {
        return graph_fail(graph, graph->state, YVEX_BACKEND_CUDA_GRAPH_REASON_CAPTURE_FAILED,
                          YVEX_ERR_BACKEND, "cuda.graph.begin",
                          "injected CUDA graph begin-capture failure", err);
    }
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuStreamBeginCapture_v2(graph->stream,
                                                               (int)graph->capture_mode),
                          "cuda.graph.begin", err);
    if (rc != YVEX_OK) {
        graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_CAPTURE_FAILED;
        return rc;
    }
    state->capture_owner = graph;
    state->capture_stream = graph->stream;
    graph->capture_origin_state = graph->state;
    graph->capture_kernel_count = 0u;
    graph->state = YVEX_BACKEND_CUDA_GRAPH_CAPTURING;
    graph->capture_started_ns = yvex_core_monotonic_ns();
    graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NONE;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Finish capture and atomically instantiate or compatibly update the executable graph.
 *
 * The backend's active capture owner and every kernel launch already submitted to its stream.
 * Commits a new graph/exec pair only after inventory, identity, and Driver admission succeed.
 * Empty or incompatible replacement capture preserves the previous executable graph. Commits
 * launch topology only; semantic/executable graph ownership remains with runtime.
 */
static int graph_end(yvex_backend_cuda_graph *graph, yvex_error *err)
{
    yvex_cuda_backend_state *state;
    yvex_backend_cuda_graph_inventory inventory;
    CUgraph candidate = NULL;
    CUgraphExec new_exec = NULL;
    cuda_kernel_binding *candidate_kernel_bindings = NULL;
    size_t candidate_kernel_count = 0u;
    CUgraphExecUpdateResultInfo update_result;
    char graph_identity[65];
    char new_exec_identity[65];
    int injected;
    int rc;
    unsigned long long now;
    unsigned long long instantiate_started;
    unsigned long long update_started;
    unsigned long long update_finished;
    size_t kernel_index;
    if (!graph || !graph->backend || graph->state != YVEX_BACKEND_CUDA_GRAPH_CAPTURING) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.end",
                            "active CUDA graph capture is required");
    }
    state = yvex_cuda_state(graph->backend);
    if (state->capture_owner != graph || state->capture_stream != graph->stream) {
        graph_mark_failed(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_INVALID_STATE);
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.end",
                            "CUDA graph capture ownership is inconsistent");
    }
    injected = graph_failure_matches("capture-end");
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuStreamEndCapture(graph->stream, &candidate),
                          "cuda.graph.end", err);
    state->capture_owner = NULL;
    state->capture_stream = NULL;
    graph->capture_count++;
    now = yvex_core_monotonic_ns();
    if (now >= graph->capture_started_ns)
        graph->capture_elapsed_ns = now - graph->capture_started_ns;
    if (rc != YVEX_OK || injected) {
        if (injected) {
            rc = graph_reject(err, YVEX_ERR_BACKEND, "cuda.graph.end",
                              "injected CUDA graph end-capture failure");
        }
        if (injected)
            graph_capture_restore(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_CAPTURE_FAILED);
        else
            graph_mark_failed(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_CAPTURE_FAILED);
        return candidate_destroy(graph, candidate, rc, err);
    }
    rc = graph_inventory(graph, candidate, &inventory, &candidate_kernel_bindings,
                         &candidate_kernel_count, graph_identity, err);
    if (rc != YVEX_OK) {
        graph_capture_restore(graph, rc == YVEX_ERR_UNSUPPORTED
                                         ? YVEX_BACKEND_CUDA_GRAPH_REASON_EMPTY_CAPTURE
                                         : YVEX_BACKEND_CUDA_GRAPH_REASON_CAPTURE_FAILED);
        return candidate_destroy(graph, candidate, rc, err);
    }
    if (graph_failure_matches("exec-identity")) {
        rc = graph_reject(err, YVEX_ERR_STATE, "cuda.graph.identity",
                          "injected CUDA graph executable identity failure");
    } else {
        rc = exec_identity(graph, graph_identity, new_exec_identity, err);
    }
    if (rc != YVEX_OK) {
        free(candidate_kernel_bindings);
        graph_capture_restore(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_CAPTURE_FAILED);
        return candidate_destroy(graph, candidate, rc, err);
    }
    if (!graph->exec) {
        instantiate_started = yvex_core_monotonic_ns();
        if (graph_failure_matches("instantiate")) {
            rc = graph_reject(err, YVEX_ERR_BACKEND, "cuda.graph.instantiate",
                              "injected CUDA graph instantiation failure");
        } else {
            rc = yvex_cuda_status(&state->driver,
                                  state->driver.cuGraphInstantiateWithFlags(&new_exec, candidate,
                                                                            0ull),
                                  "cuda.graph.instantiate", err);
        }
        if (rc != YVEX_OK) {
            free(candidate_kernel_bindings);
            graph_capture_restore(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_INSTANTIATE_FAILED);
            return candidate_destroy(graph, candidate, rc, err);
        }
        graph->graph = candidate;
        graph->exec = new_exec;
        graph->instantiate_count++;
        now = yvex_core_monotonic_ns();
        if (now >= instantiate_started)
            graph->instantiate_elapsed_ns = now - instantiate_started;
    } else {
        if (candidate_kernel_count != graph->kernel_node_count) {
            free(candidate_kernel_bindings);
            yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "cuda.graph.update",
                            "CUDA graph kernel count changed; expected=%zu actual=%zu",
                            graph->kernel_node_count, candidate_kernel_count);
            return candidate_destroy(graph, candidate, YVEX_ERR_UNSUPPORTED, err);
        }
        memset(&update_result, 0, sizeof(update_result));
        update_started = yvex_core_monotonic_ns();
        if (graph_failure_matches("update")) {
            update_result.result = 1;
            rc = YVEX_ERR_UNSUPPORTED;
        } else {
            rc = yvex_cuda_status(&state->driver,
                                  state->driver.cuGraphExecUpdate_v2(graph->exec, candidate,
                                                                    &update_result),
                                  "cuda.graph.update", err);
            if (rc == YVEX_OK && update_result.result != CUDA_GRAPH_UPDATE_SUCCESS)
                rc = YVEX_ERR_UNSUPPORTED;
        }
        update_finished = yvex_core_monotonic_ns();
        if (rc != YVEX_OK) {
            free(candidate_kernel_bindings);
            graph_capture_restore(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_UPDATE_INCOMPATIBLE);
            yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "cuda.graph.update",
                            "CUDA graph update is incompatible; update_result=%d",
                            update_result.result);
            return candidate_destroy(graph, candidate, YVEX_ERR_UNSUPPORTED, err);
        }
        for (kernel_index = 0u; kernel_index < candidate_kernel_count; ++kernel_index)
            memcpy(graph->kernel_bindings[kernel_index].identity,
                   candidate_kernel_bindings[kernel_index].identity,
                   sizeof(graph->kernel_bindings[kernel_index].identity));
        free(candidate_kernel_bindings);
        candidate_kernel_bindings = NULL;
        rc = candidate_destroy(graph, candidate, YVEX_OK, err);
        if (rc != YVEX_OK) return rc;
        graph->update_count++;
        if (update_finished >= update_started)
            graph->last_update_elapsed_ns = update_finished - update_started;
    }
    if (candidate_kernel_bindings) {
        free(graph->kernel_bindings);
        graph->kernel_bindings = candidate_kernel_bindings;
        graph->kernel_node_count = candidate_kernel_count;
    }
    graph->kernel_update_cursor = 0u;
    graph->inventory = inventory;
    memcpy(graph->launch_graph_identity, graph_identity, sizeof(graph_identity));
    memcpy(graph->graph_exec_identity, new_exec_identity, sizeof(new_exec_identity));
    graph->uploaded = 0;
    graph->state = YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED;
    graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NONE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int graph_upload(yvex_backend_cuda_graph *graph, yvex_error *err)
{
    yvex_cuda_backend_state *state;
    int rc;
    if (!graph || !graph->backend || graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED ||
        !graph->exec) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.upload",
                            "instantiated CUDA graph is required");
    }
    state = yvex_cuda_state(graph->backend);
    if (graph_failure_matches("upload")) {
        return graph_fail(graph, graph->state, YVEX_BACKEND_CUDA_GRAPH_REASON_UPLOAD_FAILED,
                          YVEX_ERR_BACKEND, "cuda.graph.upload",
                          "injected CUDA graph upload failure", err);
    }
    rc = yvex_cuda_set_current(graph->backend, "cuda.graph.upload", err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuGraphUpload(graph->exec, graph->stream),
                              "cuda.graph.upload", err);
    if (rc != YVEX_OK) {
        graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_UPLOAD_FAILED;
        return rc;
    }
    graph->uploaded = 1;
    graph->upload_count++;
    graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NONE;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Enqueue one replay of an instantiated CUDA graph executable.
 *
 * Enqueues only the captured graph on the owned stream and advances launch/replay counters. Launch
 * failure marks the object failed and publishes no successful replay counter. Launch never
 * performs CPU numerical work or an eager fallback.
 */
static int graph_launch(yvex_backend_cuda_graph *graph, yvex_error *err)
{
    yvex_cuda_backend_state *state;
    int rc;
    if (!graph || !graph->backend || graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED ||
        !graph->exec) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.launch",
                            "instantiated CUDA graph is required");
    }
    state = yvex_cuda_state(graph->backend);
    if (graph_failure_matches("launch")) {
        return graph_fail(graph, YVEX_BACKEND_CUDA_GRAPH_FAILED,
                          YVEX_BACKEND_CUDA_GRAPH_REASON_LAUNCH_FAILED,
                          YVEX_ERR_BACKEND, "cuda.graph.launch",
                          "injected CUDA graph launch failure", err);
    }
    rc = yvex_cuda_set_current(graph->backend, "cuda.graph.launch", err);
    if (rc == YVEX_OK) {
        graph->in_flight = 1;
        rc = yvex_cuda_status(&state->driver,
                              state->driver.cuGraphLaunch(graph->exec, graph->stream),
                              "cuda.graph.launch", err);
    }
    if (rc != YVEX_OK) {
        graph_mark_failed(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_LAUNCH_FAILED);
        return rc;
    }
    graph->launch_count++;
    graph->replay_count++;
    graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NONE;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Synchronize only the graph object's stream after one or more launches.
 *
 * Synchronization failure marks replay state failed and cannot be reported as completion.
 */
static int graph_synchronize(yvex_backend_cuda_graph *graph, yvex_error *err)
{
    yvex_cuda_backend_state *state;
    int rc;
    if (!graph || !graph->backend || graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED) {
        return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.synchronize",
                            "instantiated CUDA graph is required");
    }
    state = yvex_cuda_state(graph->backend);
    if (graph_failure_matches("synchronize")) {
        return graph_fail(graph, YVEX_BACKEND_CUDA_GRAPH_FAILED,
                          YVEX_BACKEND_CUDA_GRAPH_REASON_SYNCHRONIZE_FAILED,
                          YVEX_ERR_BACKEND, "cuda.graph.synchronize",
                          "injected CUDA graph synchronization failure", err);
    }
    rc = yvex_cuda_status(&state->driver, state->driver.cuStreamSynchronize(graph->stream),
                          "cuda.graph.synchronize", err);
    if (rc != YVEX_OK) {
        graph_mark_failed(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_SYNCHRONIZE_FAILED);
        return rc;
    }
    graph->in_flight = 0;
    graph->synchronize_count++;
    graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NONE;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int graph_quiesce(yvex_backend_cuda_graph *graph, yvex_error *err)
{
    yvex_cuda_backend_state *state;
    int rc;
    if (!graph->in_flight)
        return YVEX_OK;
    state = yvex_cuda_state(graph->backend);
    rc = yvex_cuda_set_current(graph->backend, "cuda.graph.quiesce", err);
    if (rc == YVEX_OK && graph_failure_matches("quiesce")) {
        rc = graph_cleanup_fail(graph, "cuda.graph.quiesce",
                                "injected CUDA graph quiescence failure", err);
    }
    if (rc == YVEX_OK)
        rc = yvex_cuda_status(&state->driver, state->driver.cuStreamSynchronize(graph->stream),
                              "cuda.graph.quiesce", err);
    if (rc != YVEX_OK) return graph_cleanup_result(graph, rc);
    graph->in_flight = 0;
    graph->synchronize_count++;
    return YVEX_OK;
}
/*
 * Abandon a failed enqueue while preserving a previously admitted executable.
 *
 * Active capture owner and the primary callback failure. Ends capture, destroys only the abandoned
 * candidate, and clears exclusive capture state. Cleanup failure replaces the callback failure
 * because ownership could not be discharged.
 */
static int graph_capture_abort(yvex_backend_cuda_graph *graph, int primary_rc, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(graph ? graph->backend : NULL);
    CUgraph abandoned = NULL;
    yvex_error primary;
    int cleanup_rc;
    int end_failed;
    if (err)
        primary = *err;
    if (!graph || !state || state->capture_owner != graph)
        return primary_rc;
    end_failed = state->driver.cuStreamEndCapture(graph->stream, &abandoned) !=
                 YVEX_CUDA_SUCCESS;
    state->capture_owner = NULL;
    state->capture_stream = NULL;
    cleanup_rc = candidate_destroy(graph, abandoned, YVEX_OK, err);
    if (cleanup_rc != YVEX_OK)
        return cleanup_rc;
    graph_capture_restore(graph, YVEX_BACKEND_CUDA_GRAPH_REASON_CAPTURE_FAILED);
    if (end_failed) {
        return graph_cleanup_fail(graph, "cuda.graph.abort",
                                  "failed CUDA graph capture could not be abandoned safely", err);
    }
    if (err)
        *err = primary;
    return primary_rc;
}
/*
 * Update one captured kernel node with current allocation-stable launch parameters.
 *
 * Backend replay owner and the next production kernel launch description. Updates only the
 * matching graph-exec node and advances its replay cursor.
 */
int yvex_cuda_graph_kernel_update(yvex_backend *backend,
                                  yvex_backend_operation_variant variant, CUfunction function,
                                  unsigned int grid, unsigned int block, unsigned int shared_bytes,
                                  void **params, const char *stage, yvex_error *err)
{
    yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    yvex_backend_cuda_graph *graph = state ? state->parameter_update_owner : NULL;
    yvex_cuda_kernel_node_params node_params;
    char identity[YVEX_SHA256_HEX_BYTES];
    int rc;
    rc = backend_dispatch_admit(backend, stage, err);
    if (rc != YVEX_OK) return rc;
    if (!graph || !graph->exec || !function || !params ||
        graph->kernel_update_cursor >= graph->kernel_node_count ||
        !state->driver.cuGraphExecKernelNodeSetParams_v2) {
        return graph_reject(err, YVEX_ERR_STATE, stage,
                            "CUDA graph kernel replay parameters are incomplete");
    }
    if (kernel_signature(backend, variant, function, grid, block, shared_bytes, stage,
                         identity, err) != YVEX_OK)
        return YVEX_ERR_STATE;
    if (strcmp(identity, graph->kernel_bindings[graph->kernel_update_cursor].identity) != 0) {
        yvex_error_setf(err, YVEX_ERR_STATE, stage,
                        "CUDA graph kernel schedule mismatch at ordinal %zu",
                        graph->kernel_update_cursor);
        return YVEX_ERR_STATE;
    }
    memset(&node_params, 0, sizeof(node_params));
    node_params.function = function;
    node_params.grid_x = grid;
    node_params.grid_y = 1u;
    node_params.grid_z = 1u;
    node_params.block_x = block;
    node_params.block_y = 1u;
    node_params.block_z = 1u;
    node_params.shared_bytes = shared_bytes;
    node_params.parameters = params;
    if (yvex_cuda_status(&state->driver,
            state->driver.cuGraphExecKernelNodeSetParams_v2(
                graph->exec, graph->kernel_bindings[graph->kernel_update_cursor].node,
                &node_params),
            stage, err) != YVEX_OK)
        return YVEX_ERR_BACKEND;
    graph->kernel_update_cursor++;
    return YVEX_OK;
}
/*
 * Capture once or replay one exact production launch sequence.
 *
 * Owns capture/instantiate/upload/replay and returns immutable graph counters. Device timing and
 * completion are explicit because either one serializes the stream. A graph may therefore enqueue
 * on the session stream and leave completion to the transaction boundary that publishes its
 * results; isolated graph streams always complete before returning.
 */
int yvex_cuda_graph_execute(yvex_backend *backend, const char *compatibility_identity,
                            yvex_cuda_graph_prepare_fn prepare,
                            yvex_cuda_graph_enqueue_fn enqueue, void *context,
                            unsigned int execution_flags, yvex_backend_cuda_graph_info *info,
                            yvex_error *err)
{
    const unsigned int allowed_flags = YVEX_CUDA_GRAPH_EXECUTION_MEASURE_DEVICE_TIME |
                                       YVEX_CUDA_GRAPH_EXECUTION_SHARED_LAUNCH_STREAM |
                                       YVEX_CUDA_GRAPH_EXECUTION_DEFER_COMPLETION;
    yvex_cuda_backend_state *state =
        backend && backend->kind == YVEX_BACKEND_KIND_CUDA ? yvex_cuda_state(backend) : NULL;
    yvex_backend_cuda_graph_options options;
    yvex_backend_cuda_graph *graph;
    int measure_device_time =
        (execution_flags & YVEX_CUDA_GRAPH_EXECUTION_MEASURE_DEVICE_TIME) != 0u;
    int shared_launch_stream =
        (execution_flags & YVEX_CUDA_GRAPH_EXECUTION_SHARED_LAUNCH_STREAM) != 0u;
    int defer_completion =
        (execution_flags & YVEX_CUDA_GRAPH_EXECUTION_DEFER_COMPLETION) != 0u;
    int created = 0;
    int rc, sync_rc, timing_rc;
    unsigned long long replay_started, replay_finished, device_elapsed = 0ull;
    yvex_error timing_error;
    if (info)
        memset(info, 0, sizeof(*info));
    if (!state || !compatibility_identity || !compatibility_identity[0] || !enqueue ||
        (execution_flags & ~allowed_flags) != 0u ||
        (defer_completion && (!shared_launch_stream || measure_device_time))) {
        return graph_reject(
            err, YVEX_ERR_INVALID_ARG, "cuda.graph.execute",
            "CUDA graph execution requires a valid backend, identity, callback, and completion policy");
    }
    rc = backend_dispatch_admit(backend, "cuda.graph.execute", err);
    if (rc != YVEX_OK) return rc;
    graph = graph_find(state, compatibility_identity, shared_launch_stream);
    if (graph && graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED &&
        graph->state != YVEX_BACKEND_CUDA_GRAPH_OPEN &&
        graph->state != YVEX_BACKEND_CUDA_GRAPH_INVALIDATED) {
        yvex_error_setf(err, YVEX_ERR_STATE, "cuda.graph.execute",
                        "cached CUDA graph cannot execute: %s",
                        graph_reason_name(graph->reason));
        return YVEX_ERR_STATE;
    }
    if (!graph) {
        memset(&options, 0, sizeof(options));
        options.schema = YVEX_BACKEND_CUDA_GRAPH_SCHEMA;
        options.capture_mode = YVEX_BACKEND_CUDA_CAPTURE_THREAD_LOCAL;
        options.compatibility_identity = compatibility_identity;
        rc = graph_open(backend, &options, shared_launch_stream, &graph, err);
        if (rc != YVEX_OK)
            return rc;
        created = 1;
    }
    if ((graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED || graph->update_requested) &&
        graph->in_flight) {
        rc = graph_quiesce(graph, err);
        if (rc != YVEX_OK) goto failed;
    }
    if (prepare) {
        state->parameter_update_owner = graph;
        rc = prepare(context, err);
        state->parameter_update_owner = NULL;
        if (rc != YVEX_OK) goto failed;
    }
    if (graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED ||
        graph->update_requested) {
        rc = graph_begin(graph, err);
        if (rc == YVEX_OK)
            rc = enqueue(context, 1, err);
        if (rc != YVEX_OK) {
            rc = graph_capture_abort(graph, rc, err);
            goto failed;
        }
        rc = graph_end(graph, err);
        if (rc == YVEX_OK)
            graph->workspace_cursor = backend->workspace_cursor;
        if (rc == YVEX_OK)
            rc = graph_upload(graph, err);
        if (rc == YVEX_OK)
            graph->update_requested = 0;
        if (rc != YVEX_OK)
            goto failed;
    } else {
        state->parameter_update_owner = graph;
        graph->kernel_update_cursor = 0u;
        rc = enqueue(context, 0, err);
        state->parameter_update_owner = NULL;
        if (rc == YVEX_OK && graph->kernel_update_cursor != graph->kernel_node_count) {
            yvex_error_setf(err, YVEX_ERR_STATE, "cuda.graph.execute",
                            "CUDA graph kernel replay schedule changed: expected=%zu actual=%zu",
                            graph->kernel_node_count, graph->kernel_update_cursor);
            rc = YVEX_ERR_STATE;
        }
        if (rc != YVEX_OK)
            return rc;
        if (backend->workspace_cursor != graph->workspace_cursor) {
            return graph_reject(err, YVEX_ERR_STATE, "cuda.graph.execute",
                                "CUDA graph replay workspace layout is not stable");
        }
    }
    replay_started = yvex_core_monotonic_ns();
    if (defer_completion) {
        graph->last_replay_elapsed_ns = 0ull;
        graph->last_device_elapsed_ns = 0ull;
    }
    rc = measure_device_time
             ? yvex_cuda_timing(backend, graph->stream, YVEX_CUDA_TIMING_BEGIN, NULL,
                                "cuda.graph.timing.begin", err)
             : YVEX_OK;
    if (rc == YVEX_OK) rc = graph_launch(graph, err);
    if (rc == YVEX_OK && !defer_completion) {
        yvex_error_clear(&timing_error);
        timing_rc = measure_device_time
                        ? yvex_cuda_timing(
                              backend, graph->stream, YVEX_CUDA_TIMING_FINISH,
                              &device_elapsed, "cuda.graph.timing.finish", &timing_error)
                        : YVEX_OK;
        sync_rc = graph_synchronize(graph, err);
        if (timing_rc != YVEX_OK) {
            if (err) *err = timing_error;
            rc = timing_rc;
        } else {
            rc = sync_rc;
        }
    } else if (measure_device_time) {
        (void)yvex_cuda_timing(backend, NULL, YVEX_CUDA_TIMING_DISCARD, NULL, NULL, NULL);
    }
    replay_finished = yvex_core_monotonic_ns();
    if (rc == YVEX_OK && !defer_completion && replay_finished >= replay_started) {
        graph->last_replay_elapsed_ns = replay_finished - replay_started;
        graph->last_device_elapsed_ns = device_elapsed;
    }
    if (rc == YVEX_OK && info)
        rc = graph_info(graph, info, err);
    return rc;
failed:
    if (created) {
        yvex_error primary = *err;
        yvex_error cleanup;
        int cleanup_rc;
        yvex_error_clear(&cleanup);
        cleanup_rc = graph_release(&graph, &cleanup);
        if (cleanup_rc != YVEX_OK) {
            *err = cleanup;
            return cleanup_rc;
        }
        *err = primary;
    }
    return rc;
}
/*
 * Bind one attention job to a stable launch-graph compatibility key.
 *
 * Missing residency, overflow, or an incomplete encoded range refuses before capture. Key admits
 * launch reuse only; it does not replace semantic/executable graph identity.
 */
int yvex_cuda_attention_graph_key(const yvex_backend *backend,
                                  const yvex_backend_attention_job *job, unsigned int first,
                                  unsigned int last, char output[160], yvex_error *err)
{
    const yvex_cuda_backend_state *state =
        backend && backend->kind == YVEX_BACKEND_KIND_CUDA ? yvex_cuda_state(backend) : NULL;
    const uintptr_t base = backend ? (uintptr_t)backend->resident_host_base : 0u;
    const yvex_cuda_attention_configuration *configuration =
        state && job
            ? yvex_cuda_attention_configuration_active(state, job->phase)
            : NULL;
    const double numeric[] = {job ? job->rms_epsilon : 0.0,
                              job ? job->position.theta : 0.0,
                              job ? job->position.scaling_factor : 0.0};
    const yvex_backend_attention_activation *activations[4];
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char hex[YVEX_SHA256_HEX_BYTES];
    size_t index;
    if (!state || !job || !output || first >= last ||
        last > YVEX_CUDA_ATTENTION_STAGE_COUNT || !base || !configuration ||
        !configuration->compatibility_identity[0]) {
        return graph_reject(
            err, YVEX_ERR_STATE, "cuda.attention.graph_key",
            "phase-bound CUDA attention residency and stage interval are required");
    }
    if (configuration->mode != YVEX_BACKEND_CUDA_ATTENTION_EAGER &&
        (job->local_count > yvex_cuda_attention_local_capacity(configuration, job, 0) ||
         job->compressed_count > configuration->compressed_capacity ||
         job->indexer_count > configuration->indexer_capacity)) {
        return graph_reject(err, YVEX_ERR_BOUNDS, "cuda.attention.graph_key",
                            "CUDA attention history exceeds its admitted capture bucket");
    }
    activations[0] = &job->attention_kv_activation;
    activations[1] = &job->compressor_activation;
    activations[2] = &job->compressor_rotated_activation;
    activations[3] = &job->indexer_query_activation;
    yvex_sha256_init(&hash);
#define HASH(value) \
    do { if (!yvex_sha256_update_u64(&hash, (unsigned long long)(value))) goto failed; } while (0)
    if (!yvex_sha256_update_text(&hash, "yvex.cuda.attention-topology.v7") ||
        !yvex_sha256_update_text(&hash, configuration->compatibility_identity) ||
        !yvex_sha256_update_text(&hash, configuration->capture_bucket))
        goto failed;
    HASH(backend->resident_generation);
    HASH(backend->workspace_generation);
    HASH(backend->host_workspace_generation);
    HASH(configuration->mode); HASH(YVEX_CUDA_ATTENTION_STAGE_COUNT);
    HASH(first); HASH(last);
    HASH(job->schema); HASH(job->phase); HASH(job->operation_scope);
    HASH(job->candidate_block_visible);
    HASH(job->token_count); HASH(job->input_stride);
    HASH(job->attention_class);
    HASH(job->evidence_level);
    HASH(job->compute_contract); HASH(job->hidden_width);
    HASH(job->q_rank); HASH(job->query_heads); HASH(job->head_dimension);
    HASH(job->kv_width); HASH(job->sliding_window); HASH(job->compression_ratio);
    HASH(job->output_groups); HASH(job->output_group_input_width); HASH(job->output_rank);
    HASH(job->indexer_heads); HASH(job->indexer_head_dimension); HASH(job->indexer_topk);
    HASH(yvex_cuda_attention_local_capacity(configuration, job, 0)); HASH(job->local_stride);
    HASH(configuration->compressed_capacity);
    HASH(job->compressed_stride);
    HASH(configuration->indexer_capacity); HASH(job->indexer_stride);
    HASH(job->main_rolling.present);
    HASH(job->main_rolling.ratio); HASH(job->main_rolling.head_dimension);
    HASH(job->main_rolling.state_width); HASH(job->main_rolling.state_slots);
    HASH(job->main_rolling.overlap);
    HASH(job->indexer_rolling.present);
    HASH(job->indexer_rolling.ratio); HASH(job->indexer_rolling.head_dimension);
    HASH(job->indexer_rolling.state_width); HASH(job->indexer_rolling.state_slots);
    HASH(job->indexer_rolling.overlap);
    HASH(job->position.original_context); HASH(job->position.beta_fast);
    HASH(job->position.beta_slow); HASH(job->position.rope_dimensions);
    for (index = 0u; index < sizeof(numeric) / sizeof(numeric[0]); ++index) {
        unsigned long long bits = 0ull;
        memcpy(&bits, &numeric[index], sizeof(numeric[index]));
        HASH(bits);
    }
    for (index = 0u; index < sizeof(activations) / sizeof(activations[0]); ++index) {
        HASH(activations[index]->required); HASH(activations[index]->block_width);
        HASH(activations[index]->quantization); HASH(activations[index]->hadamard);
    }
    for (index = 0u; index < YVEX_BACKEND_ATTENTION_WEIGHT_COUNT; ++index) {
        const yvex_backend_attention_weight *weight = &job->weights[index];
        uintptr_t encoded = (uintptr_t)weight->encoded;
        unsigned long long offset = 0ull;
        if (weight->present) {
            if (encoded < base || encoded - base > backend->resident_host_bytes ||
                weight->encoded_bytes > backend->resident_host_bytes - (encoded - base))
                goto invalid_range;
            offset = (unsigned long long)(encoded - base);
        }
        HASH(weight->present ? 1ull : 0ull); HASH(offset); HASH(weight->encoded_bytes);
        HASH(weight->row_count); HASH(weight->row_width); HASH(weight->row_bytes);
        HASH(weight->qtype);
    }
#undef HASH
    if (!yvex_sha256_final(&hash, digest))
        goto failed;
    yvex_sha256_hex(digest, hex);
    if (snprintf(output, 160u, "%s:%u-%u:%s",
                 configuration->compatibility_identity, first, last, hex) <= 0)
        goto failed;
    yvex_error_clear(err);
    return YVEX_OK;
invalid_range:
#undef HASH
    return graph_reject(err, YVEX_ERR_STATE, "cuda.attention.graph_key",
                        "CUDA attention encoded weight is outside sealed residency");
failed:
#undef HASH
    return graph_reject(err, YVEX_ERR_STATE, "cuda.attention.graph_key",
                        "CUDA attention graph compatibility serialization failed");
}
/*
 * Destroy captured graph and executable handles while retaining the reusable stream object.
 *
 * Cleanup failure preserves the still-owned handle for deterministic retry.
 */
static int graph_invalidate(yvex_backend_cuda_graph *graph, yvex_error *err)
{
    yvex_cuda_backend_state *state;
    int rc;
    if (!graph || !graph->backend) {
        return graph_reject(err, YVEX_ERR_INVALID_ARG, "cuda.graph.invalidate",
                            "attached CUDA graph object is required");
    }
    if (graph->state == YVEX_BACKEND_CUDA_GRAPH_CAPTURING) {
        return graph_fail(graph, graph->state, YVEX_BACKEND_CUDA_GRAPH_REASON_BUSY,
                          YVEX_ERR_STATE, "cuda.graph.invalidate",
                          "active CUDA graph capture must be ended before invalidation", err);
    }
    state = yvex_cuda_state(graph->backend);
    graph_poison(graph);
    rc = graph_quiesce(graph, err);
    if (rc != YVEX_OK)
        return rc;
    if (graph->exec) {
        if (graph_failure_matches("exec-destroy")) {
            return graph_cleanup_fail(graph, "cuda.graph.exec_destroy",
                                      "injected CUDA graph executable cleanup failure", err);
        }
        rc = yvex_cuda_status(&state->driver, state->driver.cuGraphExecDestroy(graph->exec),
                              "cuda.graph.exec_destroy", err);
        if (rc != YVEX_OK) return graph_cleanup_result(graph, rc);
        graph->exec = NULL;
    }
    if (graph->graph) {
        if (graph_failure_matches("graph-destroy")) {
            return graph_cleanup_fail(graph, "cuda.graph.destroy",
                                      "injected CUDA graph cleanup failure", err);
        }
        rc = yvex_cuda_status(&state->driver, state->driver.cuGraphDestroy(graph->graph),
                              "cuda.graph.destroy", err);
        if (rc != YVEX_OK) return graph_cleanup_result(graph, rc);
        graph->graph = NULL;
    }
    if (graph->pending_graph) {
        if (graph_failure_matches("graph-destroy")) {
            return graph_cleanup_fail(graph, "cuda.graph.pending_destroy",
                                      "injected pending CUDA graph cleanup failure", err);
        }
        rc = yvex_cuda_status(&state->driver, state->driver.cuGraphDestroy(graph->pending_graph),
                              "cuda.graph.pending_destroy", err);
        if (rc != YVEX_OK) return graph_cleanup_result(graph, rc);
        graph->pending_graph = NULL;
    }
    memset(&graph->inventory, 0, sizeof(graph->inventory));
    free(graph->kernel_bindings);
    graph->kernel_bindings = NULL;
    graph->kernel_node_count = 0u;
    graph->kernel_update_cursor = 0u;
    graph->launch_graph_identity[0] = '\0';
    graph->graph_exec_identity[0] = '\0';
    graph->uploaded = 0;
    graph->update_requested = 0;
    graph->state = YVEX_BACKEND_CUDA_GRAPH_INVALIDATED;
    graph->reason = YVEX_BACKEND_CUDA_GRAPH_REASON_NONE;
    graph->invalidation_count++;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int graph_info(const yvex_backend_cuda_graph *graph,
                      yvex_backend_cuda_graph_info *out, yvex_error *err)
{
    if (!graph || !out) {
        return graph_reject(err, YVEX_ERR_INVALID_ARG, "cuda.graph.info",
                            "graph and info output are required");
    }
    memset(out, 0, sizeof(*out));
    out->schema = YVEX_BACKEND_CUDA_GRAPH_SCHEMA;
    out->state = graph->state;
    out->reason = graph->reason;
    out->capture_mode = graph->capture_mode;
    out->uploaded = graph->uploaded;
    out->shared_launch_stream = !graph->stream_owned;
    out->completion_pending = graph->in_flight;
    out->inventory = graph->inventory;
    out->capture_count = graph->capture_count;
    out->instantiate_count = graph->instantiate_count;
    out->upload_count = graph->upload_count;
    out->update_count = graph->update_count;
    out->launch_count = graph->launch_count;
    out->replay_count = graph->replay_count;
    out->synchronize_count = graph->synchronize_count;
    out->invalidation_count = graph->invalidation_count;
    out->capture_elapsed_ns = graph->capture_elapsed_ns;
    out->instantiate_elapsed_ns = graph->instantiate_elapsed_ns;
    out->last_update_elapsed_ns = graph->last_update_elapsed_ns;
    out->last_replay_elapsed_ns = graph->last_replay_elapsed_ns;
    out->last_device_elapsed_ns = graph->last_device_elapsed_ns;
    memcpy(out->launch_graph_identity, graph->launch_graph_identity,
           sizeof(out->launch_graph_identity));
    memcpy(out->graph_exec_identity, graph->graph_exec_identity, sizeof(out->graph_exec_identity));
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Release one graph object after invalidating every owned Driver resource.
 *
 * Unlinks the object, destroys an owned stream, frees host ownership, and nulls the caller
 * pointer. A borrowed session stream remains owned by the backend. Cleanup failure preserves
 * ownership and pointer for retry.
 */
static int graph_release(yvex_backend_cuda_graph **graph_ptr, yvex_error *err)
{
    yvex_backend_cuda_graph *graph;
    yvex_cuda_backend_state *state;
    int rc;
    if (!graph_ptr || !*graph_ptr) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    graph = *graph_ptr;
    if (!graph->backend) {
        free(graph->capture_kernels);
        free(graph->kernel_bindings);
        free(graph->compatibility_identity);
        free(graph);
        *graph_ptr = NULL;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = graph_invalidate(graph, err);
    if (rc != YVEX_OK) return rc;
    state = yvex_cuda_state(graph->backend);
    if (graph->stream_owned) {
        if (graph_failure_matches("stream-destroy")) {
            return graph_cleanup_fail(graph, "cuda.graph.stream_destroy",
                                      "injected CUDA graph stream cleanup failure", err);
        }
        rc = yvex_cuda_status(&state->driver, state->driver.cuStreamDestroy_v2(graph->stream),
                              "cuda.graph.stream_destroy", err);
        if (rc != YVEX_OK) return graph_cleanup_result(graph, rc);
    }
    graph->stream = NULL;
    graph_unlink(graph);
    graph->backend = NULL;
    free(graph->capture_kernels);
    free(graph->compatibility_identity);
    free(graph);
    *graph_ptr = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int graph_release_registry(yvex_cuda_backend_state *state, int attention_only,
                                  unsigned long long *released, yvex_error *err)
{
    yvex_backend_cuda_graph *graph = state ? state->graphs : NULL;
    yvex_error subsequent_failure;
    int result = YVEX_OK;
    if (released) *released = 0ull;
    while (graph) {
        yvex_backend_cuda_graph *next = graph->next;
        yvex_error *release_error = result == YVEX_OK && err ? err : &subsequent_failure;
        int rc;
        if (!attention_only || graph_is_attention(state, graph)) {
            yvex_error_clear(release_error);
            rc = graph_release(&graph, release_error);
            if (rc == YVEX_OK && released) (*released)++;
            else if (rc != YVEX_OK && result == YVEX_OK) result = rc;
        }
        graph = next;
    }
    if (result == YVEX_OK) yvex_error_clear(err);
    return result;
}
int yvex_backend_cuda_attention_graph_summary_get(
    const yvex_backend *backend, yvex_backend_cuda_attention_graph_summary *out, yvex_error *err)
{
    const yvex_cuda_backend_state *state =
        backend && backend->kind == YVEX_BACKEND_KIND_CUDA ? yvex_cuda_state(backend) : NULL;
    const yvex_cuda_attention_configuration *configuration =
        state && state->attention_configuration_count &&
                state->attention_active_configuration <
                    state->attention_configuration_count
            ? &state->attention_configurations[
                  state->attention_active_configuration]
            : NULL;
    const yvex_backend_cuda_graph *graph;
    yvex_sha256 launch_hash, exec_hash;
    int have_graph = 0;
    if (!state || !out) {
        return graph_reject(err, YVEX_ERR_INVALID_ARG, "cuda.attention.graph_summary",
                            "CUDA backend and summary output are required");
    }
    memset(out, 0, sizeof(*out));
    out->schema = YVEX_BACKEND_CUDA_ATTENTION_GRAPH_SCHEMA;
    out->configured = configuration != NULL;
    out->kernel_bundle_native = state->kernel_bundle_native;
    out->selected_mode = configuration ? configuration->mode
                                       : YVEX_BACKEND_CUDA_ATTENTION_EAGER;
    out->driver_version = state->driver_version;
    if (configuration) {
        memcpy(out->compatibility_identity,
               configuration->compatibility_identity,
               sizeof(out->compatibility_identity));
        memcpy(out->capture_bucket, configuration->capture_bucket,
               sizeof(out->capture_bucket));
    }
    memcpy(out->kernel_bundle_architecture, state->kernel_bundle_architecture,
           sizeof(out->kernel_bundle_architecture));
    memcpy(out->cuda_build_identity, state->kernel_bundle_identity,
           sizeof(out->cuda_build_identity));
    yvex_sha256_init(&launch_hash);
    yvex_sha256_init(&exec_hash);
    if (!yvex_sha256_update_text(&launch_hash, "yvex.cuda.attention-graphs.v1") ||
        !yvex_sha256_update_text(&exec_hash, "yvex.cuda.attention-graph-execs.v1")) {
        return graph_reject(
            err, YVEX_ERR_STATE, "cuda.attention.graph_summary",
            "CUDA attention graph summary identity initialization failed");
    }
    for (graph = state->graphs; graph; graph = graph->next) {
        if (!graph_is_attention(state, graph))
            continue;
        out->invalidation_count += graph->invalidation_count;
        if (graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED)
            continue;
        have_graph = 1;
        out->graph_count++;
        out->capture_count += graph->capture_count;
        out->instantiate_count += graph->instantiate_count;
        out->upload_count += graph->upload_count;
        out->update_count += graph->update_count;
        out->update_pending_count += graph->update_requested ? 1ull : 0ull;
        out->replay_count += graph->replay_count;
        out->launch_count += graph->launch_count;
        out->node_count += graph->inventory.node_count;
        out->kernel_node_count += graph->inventory.kernel_node_count;
        out->memcpy_node_count += graph->inventory.memcpy_node_count;
        out->memset_node_count += graph->inventory.memset_node_count;
        out->capture_elapsed_ns += graph->capture_elapsed_ns;
        out->instantiate_elapsed_ns += graph->instantiate_elapsed_ns;
        out->last_update_elapsed_ns += graph->last_update_elapsed_ns;
        out->last_replay_elapsed_ns += graph->last_replay_elapsed_ns;
        out->last_device_elapsed_ns += graph->last_device_elapsed_ns;
        if (!yvex_sha256_update_text(&launch_hash, graph->launch_graph_identity) ||
            !yvex_sha256_update_text(&exec_hash, graph->graph_exec_identity)) {
            return graph_reject(err, YVEX_ERR_STATE, "cuda.attention.graph_summary",
                                "CUDA attention graph summary identity update failed");
        }
    }
    out->piece_count = configuration &&
                               configuration->mode ==
                                   YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE
                           ? out->graph_count : (out->graph_count ? 1ull : 0ull);
    if (have_graph &&
        (identity_finish(&launch_hash, out->launch_graph_identity, err) != YVEX_OK ||
         identity_finish(&exec_hash, out->graph_exec_identity, err) != YVEX_OK))
        return YVEX_ERR_STATE;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_backend_cuda_attention_graph_registry_count(const yvex_backend *backend,
                                                     unsigned long long *count, yvex_error *err)
{
    const yvex_cuda_backend_state *state = yvex_cuda_state(backend);
    const yvex_backend_cuda_graph *graph;
    if (!state || !count) {
        return graph_reject(err, YVEX_ERR_INVALID_ARG, "cuda.attention.graph_registry",
                            "CUDA backend and registry count output are required");
    }
    *count = 0ull;
    for (graph = state->graphs; graph; graph = graph->next)
        if (graph_is_attention(state, graph))
            (*count)++;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_backend_cuda_attention_graph_registry_get(
    const yvex_backend *backend, unsigned long long index,
    yvex_backend_cuda_attention_graph_entry *out, yvex_error *err)
{
    const yvex_cuda_backend_state *state =
        backend && backend->kind == YVEX_BACKEND_KIND_CUDA ? yvex_cuda_state(backend) : NULL;
    const yvex_backend_cuda_graph *graph;
    unsigned long long current = 0ull;
    int rc;
    if (!state || !out) {
        return graph_reject(err, YVEX_ERR_INVALID_ARG, "cuda.attention.graph_registry",
                            "CUDA backend and registry row output are required");
    }
    memset(out, 0, sizeof(*out));
    for (graph = state->graphs; graph; graph = graph->next) {
        if (!graph_is_attention(state, graph))
            continue;
        if (current++ != index)
            continue;
        if (strlen(graph->compatibility_identity) >= sizeof(out->compatibility_identity)) {
            return graph_reject(
                err, YVEX_ERR_BOUNDS, "cuda.attention.graph_registry",
                "CUDA attention graph compatibility key exceeds ABI storage");
        }
        out->schema = YVEX_BACKEND_CUDA_ATTENTION_GRAPH_SCHEMA;
        out->index = index;
        out->update_requested = graph->update_requested;
        memcpy(out->compatibility_identity, graph->compatibility_identity,
               strlen(graph->compatibility_identity) + 1u);
        rc = graph_info(graph, &out->graph, err);
        return rc;
    }
    yvex_error_setf(err, YVEX_ERR_BOUNDS, "cuda.attention.graph_registry",
                    "CUDA attention graph registry index is unavailable: %llu", index);
    return YVEX_ERR_BOUNDS;
}

int yvex_backend_cuda_attention_graph_registry_apply(
    yvex_backend *backend, yvex_backend_cuda_graph_registry_action action,
    unsigned long long *affected, yvex_error *err)
{
    yvex_cuda_backend_state *state =
        backend && backend->kind == YVEX_BACKEND_KIND_CUDA ? yvex_cuda_state(backend) : NULL;
    yvex_backend_cuda_graph *graph;
    yvex_error failure;
    unsigned long long count = 0ull;
    int result = YVEX_OK;
    int rc;
    if (!state || !affected || action < YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE ||
        action > YVEX_BACKEND_CUDA_GRAPH_REGISTRY_RELEASE) {
        return graph_reject(
            err, YVEX_ERR_INVALID_ARG, "cuda.attention.graph_registry",
            "CUDA backend, registry action, and affected-count output are required");
    }
    *affected = 0ull;
    if (action == YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE) {
        rc = backend_dispatch_admit(backend, "cuda.attention.graph_update", err);
        if (rc != YVEX_OK) return rc;
        for (graph = state->graphs; graph; graph = graph->next) {
            if (!graph_is_attention(state, graph)) continue;
            if (graph->state != YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED) {
                return graph_reject(
                    err, YVEX_ERR_STATE, "cuda.attention.graph_update",
                    "every attention graph must be instantiated before update");
            }
            count++;
        }
        if (!count) {
            return graph_reject(err, YVEX_ERR_STATE, "cuda.attention.graph_update",
                                "no attention graph is available for update");
        }
        for (graph = state->graphs; graph; graph = graph->next)
            if (graph_is_attention(state, graph)) graph->update_requested = 1;
        *affected = count;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (action == YVEX_BACKEND_CUDA_GRAPH_REGISTRY_RELEASE) {
        return graph_release_registry(state, 1, affected, err);
    }
    for (graph = state->graphs; graph; graph = graph->next) {
        if (graph_is_attention(state, graph) &&
            graph->state == YVEX_BACKEND_CUDA_GRAPH_CAPTURING) {
            return graph_fail(graph, graph->state, YVEX_BACKEND_CUDA_GRAPH_REASON_BUSY,
                              YVEX_ERR_STATE, "cuda.attention.invalidate",
                              "active attention capture prevents atomic invalidation", err);
        }
    }
    for (graph = state->graphs; graph; graph = graph->next)
        if (graph_is_attention(state, graph))
            graph_poison(graph);
    for (graph = state->graphs; graph; graph = graph->next) {
        if (!graph_is_attention(state, graph))
            continue;
        count++;
        yvex_error_clear(&failure);
        rc = graph_invalidate(graph, &failure);
        if (rc != YVEX_OK && result == YVEX_OK) {
            result = rc;
            if (err)
                *err = failure;
        }
    }
    *affected = count;
    if (result == YVEX_OK)
        yvex_error_clear(err);
    return result;
}

int yvex_cuda_graphs_close_all(yvex_backend *backend, yvex_error *err)
{
    return graph_release_registry(yvex_cuda_state(backend), 0, NULL, err);
}
