/* Own the process-lifetime catalog of source-qualified model targets. */
#include <yvex/internal/source_catalog.h>

#include <stdio.h>
#include <string.h>

static const yvex_source_target_identity source_target_identities[] = {
    {
        .target_id = YVEX_SOURCE_RELEASE_TARGET_ID,
        .family_key = YVEX_SOURCE_RELEASE_FAMILY_KEY,
        .family_display = YVEX_SOURCE_RELEASE_FAMILY_DISPLAY,
        .model_name = YVEX_SOURCE_RELEASE_NAME,
        .upstream_repo_id = YVEX_SOURCE_RELEASE_REPOSITORY,
        .source_dir_leaf = YVEX_SOURCE_RELEASE_SOURCE_LEAF,
        .upstream_revision = YVEX_SOURCE_RELEASE_REVISION,
        .upstream_index_path = YVEX_SOURCE_RELEASE_INDEX_PATH,
        .upstream_index_oid = YVEX_SOURCE_RELEASE_INDEX_OID,
        .upstream_index_size = YVEX_SOURCE_RELEASE_INDEX_SIZE,
        .upstream_inventory_authority = YVEX_SOURCE_RELEASE_INVENTORY_AUTHORITY,
        .config_model_type = YVEX_SOURCE_RELEASE_CONFIG_TYPE,
        .config_architecture = YVEX_SOURCE_RELEASE_CONFIG_ARCHITECTURE,
        .config_validation = YVEX_SOURCE_CONFIG_VALIDATION_DEEPSEEK_V4,
        .required_sidecars = YVEX_SOURCE_SIDECARS_DEEPSEEK_V4,
    },
    {
        .target_id = YVEX_SOURCE_MINIMAX_H3_TARGET_ID,
        .family_key = YVEX_SOURCE_MINIMAX_H3_FAMILY_KEY,
        .family_display = YVEX_SOURCE_MINIMAX_H3_FAMILY_DISPLAY,
        .model_name = YVEX_SOURCE_MINIMAX_H3_NAME,
        .upstream_repo_id = YVEX_SOURCE_MINIMAX_H3_REPOSITORY,
        .source_dir_leaf = YVEX_SOURCE_MINIMAX_H3_SOURCE_LEAF,
        .upstream_revision = YVEX_SOURCE_MINIMAX_H3_REVISION,
        .upstream_inventory_authority = "component-source-manifest",
        .config_validation = YVEX_SOURCE_CONFIG_VALIDATION_FAMILY_SEMANTIC,
    },
};

static const yvex_source_acquisition_target source_acquisition_targets[] = {
    {"gemma-4-e2b", "gemma", "hf", "google/gemma-4-E2B", "gemma-4-e2b", "main"},
    {"gemma-4-e2b-it", "gemma", "hf", "google/gemma-4-E2B-it", "gemma-4-e2b-it", "main"},
    {"gemma-4-e4b", "gemma", "hf", "google/gemma-4-E4B", "gemma-4-e4b", "main"},
    {"gemma-4-e4b-it", "gemma", "hf", "google/gemma-4-E4B-it", "gemma-4-e4b-it", "main"},
    {"gemma-4-12b", "gemma", "hf", "google/gemma-4-12B", "gemma-4-12b", "main"},
    {"gemma-4-12b-it", "gemma", "hf", "google/gemma-4-12B-it", "gemma-4-12b-it", "main"},
    {"gemma-4-26b-a4b", "gemma", "hf", "google/gemma-4-26B-A4B", "gemma-4-26b-a4b", "main"},
    {"gemma-4-26b-a4b-it", "gemma", "hf", "google/gemma-4-26B-A4B-it", "gemma-4-26b-a4b-it", "main"},
    {"gemma-4-31b", "gemma", "hf", "google/gemma-4-31B", "gemma-4-31b", "main"},
    {"gemma-4-31b-it", "gemma", "hf", "google/gemma-4-31B-it", "gemma-4-31b-it", "main"},
    {"qwen3-8b", "qwen", "hf", "Qwen/Qwen3-8B", "qwen3-8b", "main"},
    {"qwen3-32b", "qwen", "hf", "Qwen/Qwen3-32B", "qwen3-32b", "main"},
};

const yvex_source_target_identity *yvex_source_release_identity(void)
{
    return &source_target_identities[0];
}

const yvex_source_target_identity *yvex_source_target_identity_find(
    const char *target_id)
{
    unsigned long long index;

    if (!target_id) return NULL;
    for (index = 0ull;
         index < sizeof(source_target_identities) / sizeof(source_target_identities[0]);
         ++index)
        if (strcmp(source_target_identities[index].target_id, target_id) == 0)
            return &source_target_identities[index];
    return NULL;
}

const yvex_source_target_identity *yvex_source_target_identity_find_repository(
    const char *repository)
{
    unsigned long long index;

    if (!repository) return NULL;
    for (index = 0ull;
         index < sizeof(source_target_identities) / sizeof(source_target_identities[0]);
         ++index)
        if (strcmp(source_target_identities[index].upstream_repo_id, repository) == 0)
            return &source_target_identities[index];
    return NULL;
}

const yvex_source_acquisition_target *yvex_source_acquisition_target_find(
    const char *target_id)
{
    unsigned long long index;

    if (!target_id) return NULL;
    for (index = 0ull;
         index < sizeof(source_acquisition_targets) / sizeof(source_acquisition_targets[0]);
         ++index)
        if (strcmp(source_acquisition_targets[index].target_id, target_id) == 0)
            return &source_acquisition_targets[index];
    return NULL;
}

int yvex_source_is_release_target(const char *target_id)
{
    return target_id &&
           strcmp(target_id, source_target_identities[0].target_id) == 0;
}

int yvex_source_target_path(char *out, size_t cap, const char *models_root,
                            const yvex_source_target_identity *identity)
{
    int n;

    if (!out || cap == 0u || !models_root || !models_root[0] || !identity ||
        !identity->family_key || !identity->source_dir_leaf)
        return 0;
    n = snprintf(out, cap, "%s/hf/%s/%s", models_root, identity->family_key,
                 identity->source_dir_leaf);
    return n >= 0 && (size_t)n < cap;
}
