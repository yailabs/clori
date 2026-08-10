/*
 * Own one compilation-source lifecycle from verified source metadata through an immutable
 * transform binding and bounded payload plan. Families supply lowering policy; this owner keeps
 * source, payload, binding, and map resources out of family graph recipes.
 */
#ifndef INCLUDE_YVEX_INTERNAL_COMPILER_SOURCE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_COMPILER_SOURCE_H_INCLUDED

#include <stddef.h>
#include <yvex/core.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source_payload.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_COMPILATION_SOURCE_PROJECTION_SCHEMA_V1 1u

typedef unsigned int yvex_compilation_source_failure_code;
enum {
    YVEX_COMPILATION_SOURCE_FAILURE_NONE = 0,
    YVEX_COMPILATION_SOURCE_FAILURE_INVALID_ARGUMENT,
    YVEX_COMPILATION_SOURCE_FAILURE_SOURCE,
    YVEX_COMPILATION_SOURCE_FAILURE_SEMANTIC_MODEL,
    YVEX_COMPILATION_SOURCE_FAILURE_TRANSFORM_IR,
    YVEX_COMPILATION_SOURCE_FAILURE_LOWERING,
    YVEX_COMPILATION_SOURCE_FAILURE_MAPPING_IDENTITY,
    YVEX_COMPILATION_SOURCE_FAILURE_CONTRIBUTION,
    YVEX_COMPILATION_SOURCE_FAILURE_RANGE,
    YVEX_COMPILATION_SOURCE_FAILURE_BINDING,
    YVEX_COMPILATION_SOURCE_FAILURE_PLAN,
    YVEX_COMPILATION_SOURCE_FAILURE_ALLOCATION
};

typedef struct yvex_compilation_source_failure {
    yvex_compilation_source_failure_code code;
    unsigned long long descriptor_index, contribution_index;
    yvex_source_payload_failure payload_failure;
} yvex_compilation_source_failure;

typedef struct yvex_compilation_source_summary {
    unsigned long long mapping_identity;
    char transform_identity[YVEX_SHA256_HEX_BYTES];
    unsigned long long source_tensor_count, source_shard_count, source_header_scan_count;
    unsigned long long source_payload_bytes_read, source_lookup_count;
    unsigned long long source_collision_count, source_maximum_probe;
    unsigned long long source_snapshot_identity, descriptor_count, descriptors_covered;
    unsigned long long contribution_count, contributions_resolved, direct_contributions;
    unsigned long long fp8_weight_contributions, e8m0_scale_contributions;
    unsigned long long expert_contributions, i64_router_contributions;
    unsigned long long global_contributions, norm_contributions;
    unsigned long long shared_expert_contributions, output_head_contributions;
    unsigned long long draft_contributions, routed_expert_logical_bytes;
    unsigned long long output_head_logical_bytes, range_lookup_count;
    int complete;
} yvex_compilation_source_summary;

typedef struct yvex_compilation_source_options {
    const char *source_path, *models_root, *manifest_path;
    yvex_source_payload_budget budget;
    size_t chunk_bytes, page_bytes;
} yvex_compilation_source_options;

typedef enum {
    YVEX_COMPILATION_SOURCE_REQUIRE_DIRECT = 1u << 0,
    YVEX_COMPILATION_SOURCE_REQUIRE_FP8_WEIGHT = 1u << 1,
    YVEX_COMPILATION_SOURCE_REQUIRE_E8M0_SCALE = 1u << 2,
    YVEX_COMPILATION_SOURCE_REQUIRE_EXPERT = 1u << 3,
    YVEX_COMPILATION_SOURCE_REQUIRE_I64_ROUTER = 1u << 4,
    YVEX_COMPILATION_SOURCE_REQUIRE_GLOBAL = 1u << 5,
    YVEX_COMPILATION_SOURCE_REQUIRE_NORM = 1u << 6,
    YVEX_COMPILATION_SOURCE_REQUIRE_SHARED_EXPERT = 1u << 7,
    YVEX_COMPILATION_SOURCE_REQUIRE_OUTPUT_HEAD = 1u << 8,
    YVEX_COMPILATION_SOURCE_REQUIRE_DRAFT = 1u << 9
} yvex_compilation_source_requirement;

struct yvex_source_tensor_snapshot;
struct yvex_source_verification;
struct yvex_transform_binding;
struct yvex_transform_ir;
typedef struct yvex_artifact_lowering_map yvex_artifact_lowering_map;
typedef struct yvex_artifact_lowering_api yvex_artifact_lowering_api;

typedef struct {
    unsigned int schema_version;
    unsigned long long expected_mapping_identity, required_contribution_mask;
    const void *(*source_identity)(void);
    int (*lower)(struct yvex_transform_ir **transform,
                 yvex_artifact_lowering_map **lowering,
                 const struct yvex_source_verification *verification,
                 struct yvex_source_tensor_snapshot *snapshot,
                 yvex_compilation_source_failure *failure, yvex_error *err);
    const yvex_artifact_lowering_api *lowering;
} yvex_compilation_source_projection;

typedef struct yvex_compilation_source_session yvex_compilation_source_session;

typedef struct {
    int (*open)(yvex_compilation_source_session **out,
                const yvex_compilation_source_options *options,
                const yvex_compilation_source_projection *projection,
                yvex_compilation_source_failure *failure, yvex_error *err);
    void (*close)(yvex_compilation_source_session *session);
    const yvex_compilation_source_summary *(*summary)(
        const yvex_compilation_source_session *session);
    const struct yvex_source_verification *(*verification)(
        const yvex_compilation_source_session *session);
    const yvex_artifact_lowering_map *(*lowering)(
        const yvex_compilation_source_session *session);
    const struct yvex_transform_ir *(*transform)(
        const yvex_compilation_source_session *session);
    const struct yvex_transform_binding *(*binding)(
        const yvex_compilation_source_session *session);
    yvex_source_payload_session *(*payload)(yvex_compilation_source_session *session);
    const yvex_source_payload_plan *(*plan)(
        const yvex_compilation_source_session *session);
    const char *(*failure_name)(yvex_compilation_source_failure_code code);
} yvex_compilation_source_api;

extern const yvex_compilation_source_api yvex_compilation_source_operations;

#ifdef __cplusplus
}
#endif
#endif
