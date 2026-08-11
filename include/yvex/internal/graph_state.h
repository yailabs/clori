/*
 * Expose one bounded family-neutral state lifecycle to runtime and graph execution owners.
 *
 * State storage is derived only from sealed component recipes; commits are all-or-none. Runtime
 * retains an opaque handle while graph owns layout, mutation, identity, and cleanup.
 */
#ifndef INCLUDE_YVEX_INTERNAL_GRAPH_STATE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_GRAPH_STATE_H_INCLUDED

#include <yvex/internal/graph.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_GRAPH_ATTENTION_CAPACITY_SCHEMA_V1 1u
#define YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V1 1u
#define YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V2 2u
#define YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V3 3u
#define YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V4 4u
#define YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1 1u
#define YVEX_ATTENTION_STATE_COMPONENT_CAP 8u
#define YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1 1u
#define YVEX_ATTENTION_WORKSPACE_COMPONENT_CAP 37u

typedef enum {
    YVEX_ATTENTION_STATE_COMPONENT_HISTORY = 0,
    YVEX_ATTENTION_STATE_COMPONENT_ROLLING
} yvex_attention_state_component_kind;
typedef enum {
    YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY = 0,
    YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY,
    YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY,
    YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING,
    YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING,
    YVEX_ATTENTION_STATE_BINDING_COUNT
} yvex_attention_state_binding;
typedef struct {
    unsigned int schema_version, ordinal;
    yvex_attention_state_component_kind kind;
    yvex_attention_state_binding binding;
    unsigned long long capacity, value_width;
    yvex_attention_rolling_state_view rolling;
} yvex_attention_state_component_recipe;
struct yvex_attention_state_recipe {
    unsigned int schema_version;
    unsigned long long layer_index, selection_key, initial_position, final_position;
    unsigned int component_count;
    yvex_attention_state_component_recipe components[YVEX_ATTENTION_STATE_COMPONENT_CAP];
    char attention_plan_identity[YVEX_ATTENTION_IDENTITY_CAP];
    char identity[YVEX_ATTENTION_IDENTITY_CAP];
};
struct yvex_attention_state_recipe_request {
    unsigned long long layer_ordinal, initial_position, final_position;
    const char *attention_plan_identity;
};

/* Graph state pages are borrowed by the provider and remain owned by its page pool. */
typedef struct yvex_graph_state_page_pool yvex_graph_state_page_pool;
typedef struct yvex_graph_state_page_store yvex_graph_state_page_store;
typedef struct yvex_graph_state_bank_prefix yvex_graph_state_bank_prefix;
typedef struct {
    yvex_attention_state_component_recipe recipe;
    float *values;
    unsigned long long *positions;
    float *auxiliary;
    unsigned long long start, allocated_rows;
    yvex_graph_state_page_store *value_pages, *position_pages, *auxiliary_pages;
} yvex_graph_state_component_storage;
typedef struct {
    unsigned long long allocated_bytes, metadata_bytes, virtual_bytes;
    unsigned long long resident_bytes, page_count, resident_page_count;
    unsigned long long page_commit_count, page_release_count;
} yvex_graph_state_page_summary;
typedef struct {
    const float *values;
    const unsigned long long *positions;
    unsigned long long count, width;
} yvex_graph_state_history_span;

int yvex_graph_state_hash_u64s(yvex_sha256 *, const unsigned long long *, size_t);
int yvex_graph_state_history_project(const yvex_attention_history_view *,
    const yvex_attention_state_component_recipe *, yvex_graph_state_history_span *);
const yvex_attention_rolling_state_view *yvex_graph_state_rolling_view(
    const yvex_attention_history_view *, yvex_attention_state_binding);
int yvex_graph_state_page_pool_open(yvex_graph_state_page_pool **,
    unsigned long long, yvex_error *);
int yvex_graph_state_page_pool_bind_capacity(yvex_graph_state_page_pool *,
    const yvex_execution_capacity_plan *, yvex_error *);
void yvex_graph_state_page_pool_release(yvex_graph_state_page_pool *, unsigned long long);
int yvex_graph_state_page_pool_summary(const yvex_graph_state_page_pool *,
    yvex_graph_state_page_summary *, yvex_error *);
void yvex_graph_state_page_pool_close(yvex_graph_state_page_pool **);
int yvex_graph_state_capacity_plan_valid(const yvex_execution_capacity_plan *);
int yvex_graph_state_bank_pages_open(yvex_graph_state_page_pool *,
    const yvex_execution_capacity_plan *, const yvex_attention_summary *,
    const yvex_attention_layer_plan *, const yvex_attention_state_recipe *,
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *, yvex_error *);
int yvex_graph_state_bank_pages_reset(
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *, const yvex_attention_state_recipe *, yvex_error *);
void yvex_graph_state_bank_pages_bind(
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *, const yvex_attention_state_recipe *);
int yvex_graph_state_bank_pages_transfer(
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *, const yvex_attention_state_recipe *,
    const yvex_attention_history_view *, int, yvex_error *);
void yvex_graph_state_bank_pages_close(
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT]);
int yvex_graph_state_pages_prepare_publications(yvex_attention_history_view *,
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    const yvex_attention_state_recipe *, const yvex_attention_publication *const *,
    unsigned long long, yvex_error *);
int yvex_graph_state_pointer_table_reserve(yvex_graph_state_page_pool *, void ***,
    unsigned long long *, unsigned long long *, unsigned long long, yvex_error *);
int yvex_graph_state_initial_identity(const yvex_attention_history_view *,
    const yvex_attention_state_recipe *, const char *, char[YVEX_SHA256_HEX_CAP]);
int yvex_graph_state_advance_identity(const char *, const yvex_attention_state_recipe *,
    const char *, const yvex_attention_publication *, char[YVEX_SHA256_HEX_CAP]);
int yvex_graph_state_bank_prefix_measure(
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    const yvex_attention_state_recipe *, unsigned long long *, yvex_error *);
int yvex_graph_state_bank_prefix_capture(yvex_graph_state_bank_prefix **,
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *, const yvex_attention_state_recipe *, yvex_error *);
int yvex_graph_state_bank_prefix_compatible(const yvex_graph_state_bank_prefix *,
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    const yvex_attention_state_recipe *, yvex_error *);
int yvex_graph_state_bank_prefix_attach(const yvex_graph_state_bank_prefix *,
    yvex_graph_state_component_storage[YVEX_ATTENTION_STATE_BINDING_COUNT],
    yvex_attention_history_view *, const yvex_attention_state_recipe *, yvex_error *);
void yvex_graph_state_bank_prefix_summary(const yvex_graph_state_bank_prefix *,
    unsigned long long *, unsigned long long *, unsigned long long *, const char **);
void yvex_graph_state_bank_prefix_close(yvex_graph_state_bank_prefix **);

typedef enum {
    YVEX_ATTENTION_WORKSPACE_INGRESS = 0,
    YVEX_ATTENTION_WORKSPACE_LOCAL_VALUES,
    YVEX_ATTENTION_WORKSPACE_LOCAL_POSITIONS,
    YVEX_ATTENTION_WORKSPACE_COMPRESSED_VALUES,
    YVEX_ATTENTION_WORKSPACE_COMPRESSED_POSITIONS,
    YVEX_ATTENTION_WORKSPACE_INDEXER_VALUES,
    YVEX_ATTENTION_WORKSPACE_INDEXER_POSITIONS,
    YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_VALUES,
    YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_SCORES,
    YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_VALUES,
    YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_SCORES,
    YVEX_ATTENTION_WORKSPACE_STATE_COUNTER,
    YVEX_ATTENTION_WORKSPACE_STATUS,
    YVEX_ATTENTION_WORKSPACE_SELECTION_COUNTER,
    YVEX_ATTENTION_WORKSPACE_ENVELOPE_OUTPUT,
    YVEX_ATTENTION_WORKSPACE_Q_LOW,
    YVEX_ATTENTION_WORKSPACE_QUERY,
    YVEX_ATTENTION_WORKSPACE_RAW_KV,
    YVEX_ATTENTION_WORKSPACE_ATTENTION_VALUES,
    YVEX_ATTENTION_WORKSPACE_OUTPUT,
    YVEX_ATTENTION_WORKSPACE_ENVELOPE_STAGING,
    YVEX_ATTENTION_WORKSPACE_COMPRESSED_EMISSION,
    YVEX_ATTENTION_WORKSPACE_COMPRESSED_EMISSION_POSITION,
    YVEX_ATTENTION_WORKSPACE_INDEXER_EMISSION,
    YVEX_ATTENTION_WORKSPACE_INDEXER_EMISSION_POSITION,
    YVEX_ATTENTION_WORKSPACE_INDEX_QUERY,
    YVEX_ATTENTION_WORKSPACE_INDEX_WEIGHTS,
    YVEX_ATTENTION_WORKSPACE_TOPK_INDICES,
    YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_CANDIDATE_VALUES,
    YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_CANDIDATE_SCORES,
    YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_CANDIDATE_VALUES,
    YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_CANDIDATE_SCORES,
    YVEX_ATTENTION_WORKSPACE_CORE_INPUT_EVIDENCE,
    YVEX_ATTENTION_WORKSPACE_OUTPUT_LOW,
    YVEX_ATTENTION_WORKSPACE_TOPK_POSITIONS,
    YVEX_ATTENTION_WORKSPACE_TOPK_SCORES,
    YVEX_ATTENTION_WORKSPACE_TOPK_VALID_INDICES
} yvex_attention_workspace_component_kind;
typedef enum {
    YVEX_ATTENTION_WORKSPACE_EXECUTION = 0,
    YVEX_ATTENTION_WORKSPACE_STATE_DELTA,
    YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE
} yvex_attention_workspace_lifetime;
typedef struct {
    unsigned int schema_version, ordinal;
    yvex_attention_workspace_component_kind kind;
    yvex_attention_workspace_lifetime lifetime;
    unsigned long long element_count, element_width, alignment;
    int scales_with_tokens;
} yvex_attention_workspace_component;
struct yvex_attention_workspace_recipe {
    unsigned int schema_version;
    unsigned long long layer_index, token_capacity;
    yvex_attention_execution_mode mode;
    yvex_attention_operation_scope scope;
    yvex_attention_evidence_level evidence_level;
    unsigned int component_count;
    yvex_attention_workspace_component components[YVEX_ATTENTION_WORKSPACE_COMPONENT_CAP];
    char state_recipe_identity[YVEX_ATTENTION_IDENTITY_CAP];
    char identity[YVEX_ATTENTION_IDENTITY_CAP];
};
int yvex_attention_workspace_recipe_seal(yvex_attention_workspace_recipe *recipe,
                                         yvex_error *err);

typedef struct {
    yvex_attention_probe_scope scope;
    unsigned long long history_tokens, start_position, token_count, execution_count;
    unsigned long long layer_start, selection_key;
    int select_layer, select_selection_key, use_requested_position;
} yvex_graph_attention_capacity_request;

typedef struct {
    unsigned long long capacity, maximum_capacity;
    unsigned long long value_extent, maximum_value_extent;
} yvex_graph_attention_component_capacity;

typedef struct {
    unsigned int schema_version;
    unsigned long long layer_count, selected_layer_count, selected_binding_count, first_layer;
    unsigned long long maximum_token_count, maximum_compression_ratio;
    unsigned long long maximum_topk_capacity;
    yvex_graph_attention_component_capacity components[YVEX_ATTENTION_STATE_BINDING_COUNT];
    char attention_plan_identity[YVEX_SHA256_HEX_CAP], identity[YVEX_SHA256_HEX_CAP];
} yvex_graph_attention_capacity_summary;

typedef struct {
    unsigned long long layer_ordinal;
    int selected;
    yvex_attention_state_recipe recipe;
} yvex_graph_attention_capacity_layer;

typedef struct yvex_graph_attention_capacity_plan yvex_graph_attention_capacity_plan;
int yvex_graph_attention_capacity_plan_build_compiled(
    yvex_graph_attention_capacity_plan **out, const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layers, unsigned long long layer_count,
    const yvex_graph_attention_capacity_request *request, yvex_error *err);
int yvex_graph_attention_capacity_plan_build(
    yvex_graph_attention_capacity_plan **out, const yvex_attention_plan *attention,
    const yvex_graph_attention_capacity_request *request, yvex_error *err);
const yvex_graph_attention_capacity_summary *yvex_graph_attention_capacity_plan_summary(
    const yvex_graph_attention_capacity_plan *plan);
const yvex_graph_attention_capacity_layer *yvex_graph_attention_capacity_plan_layer(
    const yvex_graph_attention_capacity_plan *plan, unsigned long long layer_ordinal);
void yvex_graph_attention_capacity_plan_close(yvex_graph_attention_capacity_plan **plan);

typedef struct {
    unsigned long long entry_count, capacity, maximum_capacity;
} yvex_graph_attention_state_component_summary;
typedef struct {
    unsigned int schema_version;
    int sealed, persistent, cancelled, invalidated, transaction_active;
    int candidate_active, abort_required, position_consistent;
    int staged_batch_complete, prefix_selected, extension_ready;
    int paged, paging_configured;
    unsigned long long layer_count, prepared_layer_count, staged_layer_count, allocated_bytes;
    unsigned long long virtual_bytes, resident_bytes, page_table_bytes;
    unsigned long long page_count, resident_page_count, page_commit_count;
    unsigned long long page_release_count;
    unsigned long long commit_count, abort_count, cancellation_count, reset_count, generation;
    unsigned long long capacity, committed_sequence_length, next_position;
    unsigned long long staged_generation, staged_next_position, selected_prefix_count;
    yvex_graph_attention_state_component_summary components[YVEX_ATTENTION_STATE_BINDING_COUNT];
    char state_layout_identity[YVEX_SHA256_HEX_CAP], state_content_identity[YVEX_SHA256_HEX_CAP];
    char staged_state_content_identity[YVEX_SHA256_HEX_CAP];
    char capacity_plan_identity[YVEX_SHA256_HEX_CAP];
} yvex_graph_attention_state_summary;
typedef enum {
    YVEX_ATTENTION_STATE_VIEW_COMMITTED = 0,
    YVEX_ATTENTION_STATE_VIEW_CANDIDATE
} yvex_attention_state_view_kind;

#define YVEX_ATTENTION_STATE_CHECKPOINT_SCHEMA_V1 1u
typedef struct {
    unsigned int schema_version;
    unsigned long long layer_count, committed_sequence_length;
    const yvex_execution_capacity_plan *capacity;
    const yvex_attention_state_recipe *recipes;
    const yvex_attention_history_view *layers;
    const char (*layer_identities)[YVEX_SHA256_HEX_CAP];
    char state_layout_identity[YVEX_SHA256_HEX_CAP];
    char state_content_identity[YVEX_SHA256_HEX_CAP];
    char capacity_plan_identity[YVEX_SHA256_HEX_CAP];
} yvex_attention_state_checkpoint;
int yvex_attention_state_checkpoint_validate(
    const yvex_attention_state_checkpoint *checkpoint,
    const yvex_graph_attention_state_summary *provider,
    yvex_error *err);

#define YVEX_ATTENTION_STATE_PREFIX_SCHEMA_V1 1u
typedef struct yvex_attention_state_prefix yvex_attention_state_prefix;
typedef struct {
    unsigned int schema_version;
    unsigned long long layer_count, committed_sequence_length;
    unsigned long long shared_bytes, mapped_bytes, reference_count;
    char state_layout_identity[YVEX_SHA256_HEX_CAP];
    char state_content_identity[YVEX_SHA256_HEX_CAP];
    char capacity_plan_identity[YVEX_SHA256_HEX_CAP];
    char prefix_identity[YVEX_SHA256_HEX_CAP];
} yvex_attention_state_prefix_summary;
int yvex_attention_state_prefix_summary_copy(
    const yvex_attention_state_prefix *prefix,
    yvex_attention_state_prefix_summary *summary, yvex_error *err);
void yvex_attention_state_prefix_close(yvex_attention_state_prefix **prefix);

#define YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V5 5u
#define YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V6 6u
#define YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V7 7u
#define YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V8 8u
typedef struct yvex_attention_state_provider {
    unsigned int schema_version;
    void *context;
    int (*configure_pages)(void *context,
                           const yvex_execution_capacity_plan *capacity,
                           yvex_attention_failure *failure, yvex_error *err);
    int (*prepare)(void *context, unsigned long long layer_index,
                   const yvex_attention_state_recipe *recipe,
                   const yvex_attention_history_view *initial_history,
                   yvex_attention_failure *failure, yvex_error *err);
    int (*summary)(void *context, yvex_graph_attention_state_summary *out,
                   yvex_error *err);
    const yvex_execution_capacity_plan *(*capacity)(void *context);
    const yvex_attention_state_recipe *(*recipe)(
        void *context, unsigned long long layer_index);
    const yvex_attention_history_view *(*view)(
        void *context, unsigned long long layer_index,
        yvex_attention_state_view_kind kind);
    int (*identity)(void *context, unsigned long long layer_index,
                    char output[YVEX_SHA256_HEX_CAP], yvex_error *err);
    int (*begin)(void *context, unsigned long long layer_ordinal,
                 const yvex_attention_layer_plan *layer,
                 const yvex_attention_history_view *initial_history,
                 unsigned long long token_position, unsigned long long token_count,
                 const yvex_attention_cancellation *cancellation,
                 const yvex_attention_history_view **history,
                 yvex_attention_failure *failure, yvex_error *err);
    int (*stage)(void *context, const yvex_attention_publication *publication,
                 const yvex_attention_cancellation *cancellation,
                 char state_delta_identity[YVEX_SHA256_HEX_CAP],
                 yvex_attention_failure *failure, yvex_error *err);
    int (*select_prefix)(void *context, unsigned long long prefix_count,
                         unsigned long long extension_count,
                         yvex_attention_failure *failure, yvex_error *err);
    /* A coordinated commit preflights every owner before any bank is visible. A
     * successful prepare retains exclusive publication ownership until one of
     * the non-failing resolve callbacks is invoked. */
    int (*prepare_commit)(void *context, yvex_attention_failure *failure,
                          yvex_error *err);
    void (*publish_commit)(void *context);
    void (*cancel_commit)(void *context);
    int (*commit)(void *context, yvex_attention_failure *failure, yvex_error *err);
    int (*abort)(void *context, yvex_attention_failure *failure, yvex_error *err);
    int (*reset)(void *context, yvex_attention_failure *failure, yvex_error *err);
    int (*restore)(void *context,
                   const yvex_attention_state_checkpoint *checkpoint,
                   yvex_attention_failure *failure, yvex_error *err);
    int (*prefix_capture)(void *context, unsigned long long maximum_bytes,
                          yvex_attention_state_prefix **prefix,
                          yvex_attention_failure *failure, yvex_error *err);
    int (*prefix_attach)(void *context,
                         const yvex_attention_state_prefix *prefix,
                         yvex_attention_failure *failure, yvex_error *err);
    int (*invalidate)(void *context, yvex_error *err);
    int (*release)(void **context, yvex_error *err);
} yvex_attention_state_provider;
typedef struct {
    void *context;
    int (*open)(void *context, const yvex_attention_plan *plan,
                unsigned long long maximum_host_bytes,
                yvex_attention_state_provider *out,
                yvex_attention_failure *failure, yvex_error *err);
    /* Owns every candidate returned by a failed or malformed open. On success it
     * must clear candidate->context; on failure it must preserve ownership for retry. */
    int (*discard)(void *context, yvex_attention_state_provider *candidate,
                   yvex_error *err);
} yvex_attention_state_provider_factory;

int yvex_attention_state_provider_open_persistent(
    const yvex_attention_plan *plan,
    unsigned long long maximum_host_bytes, yvex_attention_state_provider *out,
    yvex_attention_failure *failure, yvex_error *err);

struct yvex_runtime_execution_session;
struct yvex_runtime_model;
struct yvex_runtime_model_failure;

typedef struct {
    void *context;
    int (*prepare)(void *context, yvex_error *err);
    void (*publish)(void *context);
    void (*cancel)(void *context);
} yvex_runtime_commit_participant;

#define YVEX_RUNTIME_STATE_PROMOTION_FACTS_SCHEMA_V1 1u
typedef struct {
    unsigned int schema_version;
    int available;
    yvex_execution_physical_facts physical;
} yvex_runtime_state_promotion_facts;

int yvex_runtime_session_prepare_persistent_state(
    struct yvex_runtime_execution_session *session,
    const yvex_graph_attention_capacity_plan *capacity,
    struct yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_session_configure_persistent_pages(
    struct yvex_runtime_execution_session *session,
    const yvex_execution_capacity_plan *capacity,
    struct yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_session_prepare_persistent_scope_state(
    struct yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    const yvex_graph_attention_capacity_plan *capacity,
    struct yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_session_reset_persistent_state(
    struct yvex_runtime_execution_session *session,
    struct yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_session_prepare_attention_probe_state(
    struct yvex_runtime_execution_session *session,
    struct yvex_runtime_model *model,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_runtime_session_prepare_attention_scope_state(
    struct yvex_runtime_execution_session *session,
    struct yvex_runtime_model *model, yvex_tensor_scope scope,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_attention_failure *failure, yvex_error *err);
int yvex_runtime_attention_probe_execute(
    struct yvex_runtime_execution_session *session,
    struct yvex_runtime_model *model,
    const yvex_attention_probe_request *request,
    yvex_attention_probe_result *result,
    struct yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_session_begin(
    struct yvex_runtime_execution_session *session,
    struct yvex_runtime_model_failure *failure, yvex_error *err);
int yvex_runtime_session_select_attention_prefix(
    struct yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    unsigned long long prefix_count, unsigned long long extension_count,
    yvex_runtime_state_promotion_facts *facts, yvex_error *err);
int yvex_runtime_session_finish(
    struct yvex_runtime_execution_session *session, int status,
    yvex_error *err);
int yvex_runtime_session_finish_scope(
    struct yvex_runtime_execution_session *session, yvex_tensor_scope scope,
    yvex_attention_transaction_disposition disposition, int status,
    yvex_error *err);
int yvex_runtime_session_finish_coordinated(
    struct yvex_runtime_execution_session *session, int status,
    const yvex_runtime_commit_participant *participant, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
