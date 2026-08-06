/*
 * Bind MiniMax-H3 component execution recipes to exact physical inputs.
 *
 * Component metadata remains insufficient until the complete file identity and independently
 * derived payload identity agree with this family-owned physical boundary. Numerical execution
 * consumes this admission later; it cannot infer source or model identity from tensor names.
 */
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/minimax_h3.h>

#include <string.h>

typedef struct {
    const char *key;
    const char *value;
} audio_metadata_fact;

static const audio_metadata_fact audio_metadata[] = {
    {"general.architecture", "minimax-h3"},
    {"general.name", "audio_vae"},
    {"yvex.logical.target", YVEX_MINIMAX_H3_TARGET_ID},
    {"yvex.logical.component", "audio_vae"},
    {"yvex.source.snapshot.identity", YVEX_MINIMAX_H3_AUDIO_SOURCE_SNAPSHOT_IDENTITY},
    {"yvex.logical.component.identity", YVEX_MINIMAX_H3_AUDIO_COMPONENT_IDENTITY},
    {"yvex.logical.component_manifest.identity",
     YVEX_MINIMAX_H3_AUDIO_COMPONENT_MANIFEST_IDENTITY},
    {"yvex.logical.architecture.identity", YVEX_MINIMAX_H3_AUDIO_ARCHITECTURE_IDENTITY},
    {"yvex.logical.role_map.identity", YVEX_MINIMAX_H3_AUDIO_ROLE_MAP_IDENTITY},
    {"yvex.logical.unresolved_requirements.identity",
     YVEX_MINIMAX_H3_AUDIO_UNRESOLVED_IDENTITY},
    {"yvex.transformation.identity", YVEX_MINIMAX_H3_AUDIO_TRANSFORM_IDENTITY},
    {"yvex.physical.profile.name", YVEX_MINIMAX_H3_AUDIO_PROFILE_NAME},
    {"yvex.physical.profile.identity", YVEX_MINIMAX_H3_AUDIO_PROFILE_IDENTITY},
    {"yvex.physical.payload_plan.identity", YVEX_MINIMAX_H3_AUDIO_PAYLOAD_PLAN_IDENTITY},
    {"yvex.payload.identity", YVEX_MINIMAX_H3_AUDIO_PAYLOAD_IDENTITY},
    {"yvex.evidence.stage", "component-artifact-planned"},
};

static const yvex_complete_artifact_admission audio_catalog = {
    .artifact_class = YVEX_ARTIFACT_CLASS_COMPONENT_YVEX,
    .metadata_count = 17ull,
    .tensor_count = YVEX_MINIMAX_H3_AUDIO_TENSORS,
    .payload_bytes = YVEX_MINIMAX_H3_AUDIO_PAYLOAD_BYTES,
    .file_bytes = YVEX_MINIMAX_H3_AUDIO_FILE_BYTES,
    .source_snapshot_identity = YVEX_MINIMAX_H3_AUDIO_SOURCE_SNAPSHOT_KEY,
    .mapping_identity = YVEX_MINIMAX_H3_AUDIO_MAPPING_IDENTITY,
    .payload_identity = YVEX_MINIMAX_H3_AUDIO_PAYLOAD_IDENTITY,
    .transform_identity = YVEX_MINIMAX_H3_AUDIO_TRANSFORM_IDENTITY,
    .profile_identity = YVEX_MINIMAX_H3_AUDIO_PROFILE_IDENTITY,
    .profile_name = YVEX_MINIMAX_H3_AUDIO_PROFILE_NAME,
    .quant_execution_identity = YVEX_MINIMAX_H3_AUDIO_QUANT_EXECUTION_IDENTITY,
    .payload_plan_identity = YVEX_MINIMAX_H3_AUDIO_PAYLOAD_PLAN_IDENTITY,
    .payload_byte_identity = YVEX_MINIMAX_H3_AUDIO_PAYLOAD_BYTE_IDENTITY,
    .writer_plan_identity = YVEX_MINIMAX_H3_AUDIO_WRITER_PLAN_IDENTITY,
    .artifact_identity = YVEX_MINIMAX_H3_AUDIO_ARTIFACT_IDENTITY,
    .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
    .logical_target = YVEX_MINIMAX_H3_TARGET_ID,
    .logical_component = "audio_vae",
    .logical_component_identity = YVEX_MINIMAX_H3_AUDIO_COMPONENT_IDENTITY,
    .native_reader_accepted = 1,
    .official_reader_accepted = 1,
    .payload_integrity_accepted = 1,
    .materialization_input_ready = 1,
};

static int audio_refuse(yvex_artifact_admission_failure *failure, const char *field,
                        unsigned long long expected, unsigned long long actual,
                        yvex_status status, yvex_error *err, const char *message)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH;
        failure->expected = expected;
        failure->actual = actual;
        yvex_core_text_copy(failure->field, sizeof(failure->field), field);
    }
    yvex_error_set(err, status, "graph.minimax_h3.audio_vae", message);
    return status;
}

static int audio_metadata_validate(const yvex_gguf *gguf,
                                   yvex_artifact_admission_failure *failure,
                                   yvex_error *err)
{
    size_t index;

    if (yvex_gguf_metadata_count(gguf) != audio_catalog.metadata_count)
        return audio_refuse(failure, "metadata-count", audio_catalog.metadata_count,
                            yvex_gguf_metadata_count(gguf), YVEX_ERR_FORMAT, err,
                            "Audio VAE artifact metadata coverage differs from the admitted file");
    for (index = 0u; index < sizeof(audio_metadata) / sizeof(audio_metadata[0]); ++index) {
        const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, audio_metadata[index].key);
        const char *text = NULL;
        unsigned long long length = 0ull;
        size_t expected = strlen(audio_metadata[index].value);

        if (!value || yvex_gguf_value_as_string(value, &text, &length) != YVEX_OK ||
            length != expected || memcmp(text, audio_metadata[index].value, expected) != 0)
            return audio_refuse(failure, audio_metadata[index].key, expected, length,
                                YVEX_ERR_FORMAT, err,
                                "Audio VAE artifact metadata identity differs from its recipe");
    }
    return YVEX_OK;
}

static int audio_tensors_validate(const yvex_tensor_table *tensors,
                                  yvex_artifact_admission_failure *failure,
                                  yvex_error *err)
{
    unsigned long long elements = 0ull;
    unsigned long long payload = 0ull;
    unsigned long long index;

    if (yvex_tensor_table_count(tensors) != audio_catalog.tensor_count)
        return audio_refuse(failure, "tensor-count", audio_catalog.tensor_count,
                            yvex_tensor_table_count(tensors), YVEX_ERR_FORMAT, err,
                            "Audio VAE tensor coverage differs from the admitted component");
    for (index = 0ull; index < yvex_tensor_table_count(tensors); ++index) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, index);
        unsigned long long tensor_elements = 1ull;
        unsigned int dimension;

        if (!tensor || tensor->ggml_type != YVEX_GGUF_QTYPE_F32 || !tensor->rank)
            return audio_refuse(failure, "tensor-qtype", YVEX_GGUF_QTYPE_F32,
                                tensor ? tensor->ggml_type : ~0ull, YVEX_ERR_FORMAT, err,
                                "Audio VAE requires the exact source-faithful F32 inventory");
        for (dimension = 0u; dimension < tensor->rank; ++dimension) {
            if (!tensor->dims[dimension] ||
                !yvex_core_u64_mul(tensor_elements, tensor->dims[dimension], &tensor_elements))
                return audio_refuse(failure, "tensor-elements", 1ull, 0ull,
                                    YVEX_ERR_BOUNDS, err,
                                    "Audio VAE tensor element accounting overflowed");
        }
        if (!yvex_core_u64_add(elements, tensor_elements, &elements) ||
            !yvex_core_u64_add(payload, tensor->storage_bytes, &payload))
            return audio_refuse(failure, "tensor-population", 1ull, 0ull,
                                YVEX_ERR_BOUNDS, err,
                                "Audio VAE aggregate tensor accounting overflowed");
    }
    if (elements != YVEX_MINIMAX_H3_AUDIO_ELEMENTS)
        return audio_refuse(failure, "element-count", YVEX_MINIMAX_H3_AUDIO_ELEMENTS,
                            elements, YVEX_ERR_FORMAT, err,
                            "Audio VAE aggregate element count differs from its recipe");
    if (payload != audio_catalog.payload_bytes)
        return audio_refuse(failure, "payload-bytes", audio_catalog.payload_bytes,
                            payload, YVEX_ERR_FORMAT, err,
                            "Audio VAE aggregate payload extent differs from its recipe");
    return YVEX_OK;
}

static int audio_vae_admit(const yvex_artifact *artifact, const yvex_gguf *gguf,
                           const yvex_tensor_table *tensors,
                           yvex_complete_artifact_admission *out,
                           yvex_artifact_admission_failure *failure, yvex_error *err)
{
    int rc;

    if (!artifact || !gguf || !tensors || !out)
        return audio_refuse(failure, "arguments", 4ull, 0ull, YVEX_ERR_INVALID_ARG,
                            err, "Audio VAE admission requires artifact and structural views");
    rc = audio_metadata_validate(gguf, failure, err);
    if (rc == YVEX_OK) rc = audio_tensors_validate(tensors, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admit_component(artifact, &audio_catalog, out, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(artifact, out, NULL, NULL, failure, err);
    return rc;
}

const yvex_minimax_h3_graph_api *yvex_graph_register_minimax_h3(void)
{
    static const yvex_minimax_h3_graph_api api = {audio_vae_admit};

    return &api;
}
