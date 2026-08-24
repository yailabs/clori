/*
 * Model admission consumes artifact proofs through these typed gates and reports. The interface
 * cannot promote an incomplete artifact into runtime capability.
 */
#ifndef INCLUDE_YVEX_INTERNAL_MODEL_ARTIFACT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MODEL_ARTIFACT_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/core.h>
#include <yvex/registry.h>
#include <yvex/internal/artifact.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model-facing artifact gates. */
typedef struct {
    yvex_model_gate_status status;
    yvex_model_support_level support_level;
    const char *artifact_identity;
    const char *artifact_path;
    const char *profile_name;
    unsigned long long tensor_count;
    unsigned long long file_bytes;
    int complete_artifact_admitted;
    int materialization_input_ready;
    int execution_ready;
} yvex_model_complete_artifact_gate_fact;
int yvex_model_artifact_gate_from_admission(
    const yvex_complete_artifact_admission *admission,
    yvex_model_complete_artifact_gate_fact *fact,
    yvex_error *err);

typedef struct {
    char *alias;
    char *family;
    char *model;
    char *scope;
    char *artifact_class;
    char *qprofile;
    char *calibration;
    char *producer;
    char *schema_version;
    char *path;
    char *sha256;
    unsigned long long file_size;
    char *format;
    char *architecture;
    unsigned long long tensor_count;
    unsigned long long known_tensor_bytes;
    char *primary_tensor_name;
    char *primary_tensor_role;
    char *primary_tensor_dtype;
    unsigned int primary_tensor_rank;
    char *primary_tensor_dims;
    unsigned long long primary_tensor_bytes;
    char *support_level;
    int selected_embedding_ready;
    unsigned long long selected_embedding_hidden_size;
    unsigned long long selected_embedding_vocab_size;
    unsigned long long selected_embedding_output_count;
    unsigned long long selected_embedding_slice_bytes;
    int execution_ready;
    char *runtime_profile;
    char *runtime_installation;
    char *runtime_binding;
    char *runtime_target;
    char *runtime_backend;
    char *runtime_mode;
    unsigned long long runtime_context;
} yvex_model_registry_owned_entry;
struct yvex_model_registry {
    yvex_model_registry_owned_entry *entries;
    unsigned long long count;
    unsigned long long cap;
};

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_MODEL_ARTIFACT_H_INCLUDED */
