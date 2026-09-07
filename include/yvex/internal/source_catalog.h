/* Immutable source-qualified target truth shared without importing family implementations. */
#ifndef INCLUDE_YVEX_INTERNAL_SOURCE_CATALOG_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SOURCE_CATALOG_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    YVEX_SOURCE_CONFIG_VALIDATION_DEEPSEEK_V4 = 0,
    YVEX_SOURCE_CONFIG_VALIDATION_FAMILY_SEMANTIC = 1
} yvex_source_config_validation;

enum {
    YVEX_SOURCE_SIDECAR_CONFIG = 1u << 0,
    YVEX_SOURCE_SIDECAR_TOKENIZER = 1u << 1,
    YVEX_SOURCE_SIDECAR_TOKENIZER_CONFIG = 1u << 2,
    YVEX_SOURCE_SIDECAR_GENERATION_CONFIG = 1u << 3,
    YVEX_SOURCE_SIDECAR_INFERENCE_CONFIG = 1u << 4,
    YVEX_SOURCE_SIDECARS_TEXT = YVEX_SOURCE_SIDECAR_CONFIG |
                                 YVEX_SOURCE_SIDECAR_TOKENIZER |
                                 YVEX_SOURCE_SIDECAR_TOKENIZER_CONFIG |
                                 YVEX_SOURCE_SIDECAR_GENERATION_CONFIG,
    YVEX_SOURCE_SIDECARS_DEEPSEEK_V4 = YVEX_SOURCE_SIDECARS_TEXT |
                                        YVEX_SOURCE_SIDECAR_INFERENCE_CONFIG
};

/* A source-qualified logical relation does not grant execution capability.
 * Legacy selectors are exact tuples, never prefixes or filesystem guesses. */
typedef struct {
    const char *identity, *family, *model, *display_name;
    const char *family_aliases[2], *model_aliases[2];
    const char *related_repository, *related_revision;
} yvex_source_logical_model;

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
    yvex_source_config_validation config_validation;
    unsigned int required_sidecars;
    const yvex_source_logical_model *logical_model;
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

#define YVEX_SOURCE_QWEN3_8_27B_TARGET_ID "qwen3.8-27b"
#define YVEX_SOURCE_QWEN3_8_27B_FAMILY_KEY "qwen"
#define YVEX_SOURCE_QWEN3_8_27B_FAMILY_DISPLAY "Qwen3.5"
#define YVEX_SOURCE_QWEN3_8_27B_NAME "Qwen3.8-27B"
#define YVEX_SOURCE_QWEN3_8_27B_REPOSITORY "Qwen/Qwen3.8-27B"
#define YVEX_SOURCE_QWEN3_8_27B_SOURCE_LEAF "qwen3.8-27b"
#define YVEX_SOURCE_QWEN3_8_27B_REVISION \
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
#define YVEX_SOURCE_QWEN3_8_27B_INDEX_PATH "model.safetensors.index.json"
#define YVEX_SOURCE_QWEN3_8_27B_INDEX_OID \
    "da35e3c564457dface7d138f0b6cac284ff8958c"
#define YVEX_SOURCE_QWEN3_8_27B_INDEX_SIZE 112216ull
#define YVEX_SOURCE_QWEN3_8_27B_CONFIG_TYPE "qwen3_5"
#define YVEX_SOURCE_QWEN3_8_27B_CONFIG_ARCHITECTURE \
    "Qwen3_5ForConditionalGeneration"

const yvex_source_target_identity *yvex_source_release_identity(void);
const yvex_source_target_identity *yvex_source_target_identity_find(
    const char *target_id);
const yvex_source_target_identity *yvex_source_target_identity_find_repository(
    const char *repository);
const yvex_source_acquisition_target *yvex_source_acquisition_target_find(
    const char *target_id);
const yvex_source_logical_model *yvex_source_logical_model_for_registry(
    const char *family, const char *model);
const yvex_source_logical_model *yvex_source_logical_model_for_revision(
    const char *provider, const char *repository, const char *revision);
int yvex_source_is_release_target(const char *target_id);
int yvex_source_target_path(char *out, size_t cap, const char *models_root,
                            const yvex_source_target_identity *identity);
/* Managed provider bytes are addressed by repository and immutable revision,
 * independently of family classification or the user's display alias. */
int yvex_source_provider_path(char *out, size_t cap, const char *models_root,
                              const char *repository, const char *revision);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_SOURCE_CATALOG_H_INCLUDED */
