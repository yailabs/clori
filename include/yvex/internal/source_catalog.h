/* Immutable source-qualified target truth shared without importing family implementations. */
#ifndef INCLUDE_YVEX_INTERNAL_SOURCE_CATALOG_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SOURCE_CATALOG_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *target_id;
    const char *family_key;
    const char *family_display;
    const char *model_name;
    const char *upstream_repo_id;
    const char *source_dir_leaf;
    const char *upstream_revision;
    const char *upstream_index_path;
    const char *upstream_index_oid;
    unsigned long long upstream_index_size;
    const char *upstream_inventory_authority;
    const char *config_model_type;
    const char *config_architecture;
} yvex_source_target_identity;

/*
 * Acquisition routing is colder and weaker than a qualified target identity: the default
 * reference may be mutable and must be resolved to immutable provider evidence before use.
 */
typedef struct {
    const char *target_id;
    const char *family_key;
    const char *provider;
    const char *repository;
    const char *source_dir_leaf;
    const char *default_reference;
} yvex_source_acquisition_target;

#define YVEX_SOURCE_RELEASE_TARGET_ID "deepseek4-v4-flash-dspark"
#define YVEX_SOURCE_RETIRED_TARGET_ID "deepseek4-v4-flash"
#define YVEX_SOURCE_RELEASE_FAMILY_KEY "deepseek"
#define YVEX_SOURCE_RELEASE_FAMILY_DISPLAY "DeepSeek"
#define YVEX_SOURCE_RELEASE_NAME "DeepSeek-V4-Flash-DSpark"
#define YVEX_SOURCE_RELEASE_REPOSITORY "deepseek-ai/DeepSeek-V4-Flash-DSpark"
#define YVEX_SOURCE_RELEASE_SOURCE_LEAF "DeepSeek-V4-Flash-DSpark"
#define YVEX_SOURCE_RELEASE_MANIFEST_LEAF \
    "deepseek-v4-flash-dspark-source-manifest.json"
#define YVEX_SOURCE_RELEASE_REVISION \
    "62af8fffb2f7030cac4de2f0169f5b8d1101b646"
#define YVEX_SOURCE_RELEASE_INDEX_PATH "model.safetensors.index.json"
#define YVEX_SOURCE_RELEASE_INDEX_OID \
    "c3b10d45a829545fbf0d9d2880a1aa0b9ab3b43a"
#define YVEX_SOURCE_RELEASE_INDEX_SIZE 5602871ull
#define YVEX_SOURCE_RELEASE_INVENTORY_AUTHORITY "upstream-index"
#define YVEX_SOURCE_RELEASE_CONFIG_TYPE "deepseek_v4"
#define YVEX_SOURCE_RELEASE_CONFIG_ARCHITECTURE "DeepseekV4ForCausalLM"

#define YVEX_SOURCE_MINIMAX_H3_TARGET_ID "minimax-h3-fl2va"
#define YVEX_SOURCE_MINIMAX_H3_FAMILY_KEY "minimax-h3"
#define YVEX_SOURCE_MINIMAX_H3_FAMILY_DISPLAY "MiniMax-H3"
#define YVEX_SOURCE_MINIMAX_H3_NAME "MiniMax-H3 Base FL2VA"
#define YVEX_SOURCE_MINIMAX_H3_REPOSITORY "MiniMaxAI/MiniMax-H3"
#define YVEX_SOURCE_MINIMAX_H3_SOURCE_LEAF "MiniMax-H3"
#define YVEX_SOURCE_MINIMAX_H3_REVISION \
    "b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08"

const yvex_source_target_identity *yvex_source_release_identity(void);
const yvex_source_target_identity *yvex_source_target_identity_find(
    const char *target_id);
const yvex_source_target_identity *yvex_source_target_identity_find_repository(
    const char *repository);
const yvex_source_acquisition_target *yvex_source_acquisition_target_find(
    const char *target_id);
int yvex_source_is_release_target(const char *target_id);
int yvex_source_target_path(char *out, size_t cap, const char *models_root,
                            const yvex_source_target_identity *identity);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_SOURCE_CATALOG_H_INCLUDED */
