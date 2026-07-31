/* Owner: model.artifacts.internal (model.artifacts).
 * Owns: model-artifact references, reports, gates, and state publication.
 * Does not own: artifact byte ownership, runtime, or rendering.
 * Invariants: declarations have one owner, stable ordering, and no hidden capability promotion.
 * Boundary: model-facing artifact control.
 * Purpose: provide the canonical model-facing artifact control contract.
 * Inputs: typed immutable facts and explicitly owned mutable lifecycle objects.
 * Effects: only declared lifecycle, allocation, I/O, and publication operations mutate state.
 * Failure: typed refusals leave outputs defined and preserve caller-owned state. */
#ifndef INCLUDE_YVEX_INTERNAL_MODEL_ARTIFACT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_MODEL_ARTIFACT_H_INCLUDED

#include <yvex/artifact.h>
#include <yvex/core.h>
#include <yvex/registry.h>
#include <yvex/internal/artifact.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Gate contract. */
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

/* Private contract. */
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
} yvex_model_registry_owned_entry;
struct yvex_model_registry {
    yvex_model_registry_owned_entry *entries;
    unsigned long long count;
    unsigned long long cap;
};

/* Write contract. */
int yvex_model_registry_write_json_file(const yvex_model_registry *registry,
                                        const char *path,
                                        yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_MODEL_ARTIFACT_H_INCLUDED */
