/* Pure Mamba2 source semantics. No session, backend placement or execution claim lives here. */
#ifndef INCLUDE_YVEX_INTERNAL_FAMILIES_MAMBA2_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_FAMILIES_MAMBA2_H_INCLUDED

#include <yvex/internal/source.h>
#include <yvex/internal/semantic_decoder.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_MAMBA2_TARGET "mamba-codestral-7b-v0.1"
#define YVEX_MAMBA2_REPOSITORY "mistralai/Mamba-Codestral-7B-v0.1"
#define YVEX_MAMBA2_REVISION "4f086c08c1e0f07bdc50ca25125dbbf7475d21da"
#define YVEX_MAMBA2_LAYER_CAP 256u

typedef enum {
    YVEX_MAMBA2_ROLE_UNKNOWN = 0,
    YVEX_MAMBA2_ROLE_EMBEDDING,
    YVEX_MAMBA2_ROLE_FINAL_NORM,
    YVEX_MAMBA2_ROLE_LM_HEAD,
    YVEX_MAMBA2_ROLE_BLOCK_NORM,
    YVEX_MAMBA2_ROLE_INPUT_PROJECTION,
    YVEX_MAMBA2_ROLE_CONVOLUTION_WEIGHT,
    YVEX_MAMBA2_ROLE_CONVOLUTION_BIAS,
    YVEX_MAMBA2_ROLE_DECAY_LOG,
    YVEX_MAMBA2_ROLE_SKIP,
    YVEX_MAMBA2_ROLE_TIME_BIAS,
    YVEX_MAMBA2_ROLE_GATED_NORM,
    YVEX_MAMBA2_ROLE_OUTPUT_PROJECTION,
    YVEX_MAMBA2_ROLE_COUNT
} yvex_mamba2_role;

typedef struct {
    unsigned long long hidden_size, layer_count, vocabulary_size, expansion;
    unsigned long long chunk_size;
    yvex_selective_ssd_geometry mixer;
    unsigned long long config_bos, config_eos, config_pad;
    unsigned long long generation_bos, generation_eos, generation_pad;
    unsigned long long tokenizer_bos, tokenizer_eos, tokenizer_unk;
    double normalization_epsilon;
    int residual_in_f32, declared_norm_before_gate, token_policy_conflict;
    int normalization_policy_conflict, architecture_complete;
    char source_revision[65], architecture_identity[65];
} yvex_mamba2_architecture;

typedef struct {
    yvex_mamba2_role role;
    unsigned long long layer_index;
} yvex_mamba2_tensor_binding;

typedef struct {
    unsigned long long tensors, tensor_bytes, required_tensors;
    unsigned long long role_counts[YVEX_MAMBA2_ROLE_COUNT];
    char role_identity[65];
    int complete;
} yvex_mamba2_inventory;

typedef struct {
    unsigned int schema_version;
    int (*open)(const yvex_source_verification *, yvex_mamba2_architecture *, yvex_error *);
    int (*tensor_classify)(const yvex_mamba2_architecture *, const yvex_native_weight_info *,
                          yvex_mamba2_tensor_binding *, yvex_error *);
    int (*tensor_audit)(const yvex_mamba2_architecture *, const yvex_native_weight_table *,
                       yvex_mamba2_inventory *, yvex_error *);
    int (*snapshot_audit)(const yvex_mamba2_architecture *, const yvex_source_tensor_snapshot *,
                         yvex_mamba2_inventory *, yvex_error *);
    const char *(*role_name)(yvex_mamba2_role);
} yvex_mamba2_api;

const yvex_mamba2_api *yvex_model_register_mamba2(void);

#ifdef __cplusplus
}
#endif
#endif
