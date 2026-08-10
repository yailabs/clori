/*
 * Verify exact serialized artifacts and bind their independent proofs.
 *
 * Roundtrip hashes every byte once; complete admission never upgrades a proof, external GGUF,
 * incomplete file, or changed file snapshot. Native verification feeds complete-artifact
 * admission; acceptance means ready for materialization only.
 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>
#define ROUNDTRIP_DEFAULT_CHUNK (8u * 1024u * 1024u)
static const char *const artifact_admission_names[] = {
    [YVEX_ARTIFACT_ADMISSION_OK] = "ok", [YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT] = "invalid-argument",
    [YVEX_ARTIFACT_ADMISSION_WRITER_INCOMPLETE] = "writer-incomplete",
    [YVEX_ARTIFACT_ADMISSION_EMISSION_INCOMPLETE] = "emission-incomplete",
    [YVEX_ARTIFACT_ADMISSION_NATIVE_ROUNDTRIP] = "native-roundtrip",
    [YVEX_ARTIFACT_ADMISSION_OFFICIAL_READER] = "official-reader",
    [YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH] = "identity-mismatch",
    [YVEX_ARTIFACT_ADMISSION_TOKENIZER_INCOMPLETE] = "tokenizer-incomplete",
    [YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE] = "tensor-coverage", [YVEX_ARTIFACT_ADMISSION_FILE_OPEN] = "file-open",
    [YVEX_ARTIFACT_ADMISSION_FILE_DRIFT] = "file-drift",
};

static void artifact_identity_copy(char destination[YVEX_SHA256_HEX_CAP], const char *source)
{
    yvex_core_text_copy(destination, YVEX_SHA256_HEX_CAP, source);
}

static const char *artifact_name_at(const char *const *names, size_t count, unsigned int code,
                                    const char *fallback)
{
    return code < count && names[code] ? names[code] : fallback;
}

static int admission_fail(yvex_artifact_admission_failure *failure,
                          yvex_artifact_admission_code code, const char *field,
                          unsigned long long expected, unsigned long long actual, yvex_error *err,
                          yvex_status status, const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->expected = expected;
        failure->actual = actual;
        if (field)
            yvex_core_text_copy(failure->field, sizeof(failure->field), field);
    }
    yvex_error_set(err, status, "artifact.complete.admission", message);
    return status;
}

static int admission_snapshot_matches(const yvex_artifact_snapshot *snapshot,
                                      const yvex_gguf_file_sink_summary *emission) {
    return snapshot && emission && snapshot->device == emission->file_device &&
           snapshot->inode == emission->file_inode && snapshot->size == emission->file_size &&
           snapshot->mtime_seconds == emission->published_mtime_seconds &&
           snapshot->mtime_nanoseconds == emission->published_mtime_nanoseconds &&
           snapshot->ctime_seconds == emission->published_ctime_seconds &&
           snapshot->ctime_nanoseconds == emission->published_ctime_nanoseconds;
}

static int admission_official_snapshot_matches(const yvex_artifact_official_reader_fact *official,
                                               const yvex_gguf_file_sink_summary *emission) {
    return official && emission && official->file_device == emission->file_device &&
           official->file_inode == emission->file_inode &&
           official->file_bytes == emission->file_size &&
           official->file_mtime_seconds == emission->validated_mtime_seconds &&
           official->file_mtime_nanoseconds == emission->validated_mtime_nanoseconds &&
           official->file_ctime_seconds == emission->validated_ctime_seconds &&
           official->file_ctime_nanoseconds == emission->validated_ctime_nanoseconds;
}

/* Bind all complete-artifact admission evidence into one immutable record identity. */
static int admission_identity_encode(const yvex_complete_artifact_admission *admission,
                                     char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];

    if (!admission || !output)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, admission->artifact_class == YVEX_ARTIFACT_CLASS_COMPONENT_YVEX
                       ? "yvex.component-artifact.admission.v1"
                       : "yvex.artifact.admission.v2") ||
        !yvex_sha256_update_u64(&hash, admission->artifact_class) ||
        !yvex_sha256_update_u64(&hash, admission->metadata_count) ||
        !yvex_sha256_update_u64(&hash, admission->tensor_count) ||
        !yvex_sha256_update_u64(&hash, admission->payload_bytes) ||
        !yvex_sha256_update_u64(&hash, admission->file_bytes) ||
        !yvex_sha256_update_u64(&hash, admission->source_snapshot_identity) ||
        !yvex_sha256_update_u64(&hash, admission->mapping_identity) ||
        !yvex_sha256_update_text(&hash, admission->payload_identity) ||
        !yvex_sha256_update_text(&hash, admission->transform_identity) ||
        !yvex_sha256_update_text(&hash, admission->profile_identity) ||
        !yvex_sha256_update_text(&hash, admission->profile_name) ||
        !yvex_sha256_update_text(&hash, admission->quant_execution_identity) ||
        !yvex_sha256_update_text(&hash, admission->payload_plan_identity) ||
        !yvex_sha256_update_text(&hash, admission->payload_byte_identity) ||
        !yvex_sha256_update_text(&hash, admission->writer_plan_identity) ||
        !yvex_sha256_update_text(&hash, admission->artifact_identity) ||
        !yvex_sha256_update_text(&hash, admission->official_reader_revision) ||
        (admission->artifact_class == YVEX_ARTIFACT_CLASS_COMPONENT_YVEX &&
         (!yvex_sha256_update_text(&hash, admission->logical_target) ||
          !yvex_sha256_update_text(&hash, admission->logical_component) ||
          !yvex_sha256_update_text(&hash, admission->logical_component_identity))) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int admission_complete_catalog_valid(
    const yvex_complete_artifact_admission *admission)
{
    return admission &&
           admission->artifact_class == YVEX_ARTIFACT_CLASS_COMPLETE_YVEX &&
           admission->metadata_count && admission->tensor_count &&
           admission->payload_bytes && admission->file_bytes &&
           admission->source_snapshot_identity && admission->mapping_identity &&
           yvex_sha256_hex_is_valid(admission->artifact_identity) &&
           yvex_sha256_hex_is_valid(admission->profile_identity) &&
           yvex_sha256_hex_is_valid(admission->quant_execution_identity) &&
           yvex_sha256_hex_is_valid(admission->payload_plan_identity) &&
           yvex_sha256_hex_is_valid(admission->payload_byte_identity) &&
           yvex_sha256_hex_is_valid(admission->payload_identity) &&
           yvex_sha256_hex_is_valid(admission->transform_identity) &&
           yvex_sha256_hex_is_valid(admission->writer_plan_identity) &&
           strcmp(admission->official_reader_revision,
                  YVEX_GGUF_OFFICIAL_READER_REVISION) == 0 &&
           admission->tokenizer_complete && admission->native_reader_accepted &&
           admission->official_reader_accepted &&
           admission->payload_integrity_accepted &&
           admission->materialization_input_ready;
}

static int admission_component_catalog_valid(
    const yvex_complete_artifact_admission *catalog)
{
    return catalog && catalog->artifact_class == YVEX_ARTIFACT_CLASS_COMPONENT_YVEX &&
           catalog->metadata_count && catalog->tensor_count && catalog->payload_bytes &&
           catalog->file_bytes && catalog->source_snapshot_identity &&
           catalog->mapping_identity && catalog->logical_target[0] &&
           catalog->logical_component[0] &&
           yvex_sha256_hex_is_valid(catalog->logical_component_identity) &&
           yvex_sha256_hex_is_valid(catalog->artifact_identity) &&
           yvex_sha256_hex_is_valid(catalog->profile_identity) &&
           yvex_sha256_hex_is_valid(catalog->quant_execution_identity) &&
           yvex_sha256_hex_is_valid(catalog->payload_plan_identity) &&
           yvex_sha256_hex_is_valid(catalog->payload_byte_identity) &&
           yvex_sha256_hex_is_valid(catalog->payload_identity) &&
           yvex_sha256_hex_is_valid(catalog->transform_identity) &&
           yvex_sha256_hex_is_valid(catalog->writer_plan_identity) &&
           strcmp(catalog->official_reader_revision,
                  YVEX_GGUF_OFFICIAL_READER_REVISION) == 0 &&
           !catalog->tokenizer_complete && catalog->native_reader_accepted &&
           catalog->official_reader_accepted && catalog->payload_integrity_accepted &&
           catalog->materialization_input_ready && !catalog->runtime_supported;
}

int yvex_complete_artifact_admit(const yvex_artifact_admission_request *request,
                                 yvex_complete_artifact_admission *out,
                                 yvex_artifact_admission_failure *failure, yvex_error *err) {
    const yvex_gguf_writer_plan_summary *plan;
    const yvex_gguf_file_sink_summary *emission;
    const yvex_gguf_roundtrip_summary *roundtrip;
    const yvex_artifact_official_reader_fact *official;
    yvex_artifact_options artifact_options;
    yvex_artifact *artifact = NULL;
    yvex_artifact_snapshot snapshot;
    int rc;

    if (out)
        memset(out, 0, sizeof(*out));
    if (!request || !out || !request->artifact_path || !request->artifact_path[0] ||
        !request->writer_plan || !request->emission || !request->native_roundtrip ||
        !request->official_reader)
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT, "request", 1u, 0u,
                              err, YVEX_ERR_INVALID_ARG,
                              "complete artifact admission request is incomplete");
    plan = yvex_gguf_writer_plan_summary_get(request->writer_plan);
    emission = request->emission;
    roundtrip = request->native_roundtrip;
    official = request->official_reader;
    if (!plan || !plan->complete)
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_WRITER_INCOMPLETE, "writer", 1u, 0u,
                              err, YVEX_ERR_STATE, "sealed complete writer plan is required");
    if (!emission->finalized || !emission->published ||
        emission->committed_terminals != plan->tensor_count || emission->aborted_terminals != 0u ||
        emission->encoded_bytes_written != plan->tensor_payload_bytes ||
        emission->file_size != plan->final_file_bytes)
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_EMISSION_INCOMPLETE, "emission",
                              plan->tensor_count, emission->committed_terminals, err,
                              YVEX_ERR_STATE,
                              "published emission is incomplete or disagrees with its plan");
    if (!roundtrip->complete || !roundtrip->reader_accepted || !roundtrip->layout_accepted ||
        !roundtrip->payload_accepted || !roundtrip->snapshot_stable ||
        roundtrip->bytes_hashed != plan->final_file_bytes ||
        roundtrip->terminals_verified != plan->tensor_count ||
        roundtrip->file_device != emission->file_device ||
        roundtrip->file_inode != emission->file_inode ||
        roundtrip->file_mtime_seconds != emission->validated_mtime_seconds ||
        roundtrip->file_mtime_nanoseconds != emission->validated_mtime_nanoseconds ||
        roundtrip->file_ctime_seconds != emission->validated_ctime_seconds ||
        roundtrip->file_ctime_nanoseconds != emission->validated_ctime_nanoseconds)
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_NATIVE_ROUNDTRIP, "native-roundtrip",
                              plan->final_file_bytes, roundtrip->bytes_hashed, err, YVEX_ERR_FORMAT,
                              "native full-file roundtrip is incomplete");
    if (!official->accepted ||
        strcmp(official->revision, YVEX_GGUF_OFFICIAL_READER_REVISION) != 0 ||
        official->metadata_count != plan->metadata_count ||
        official->tensor_count != plan->tensor_count ||
        official->file_bytes != plan->final_file_bytes ||
        !admission_official_snapshot_matches(official, emission))
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_OFFICIAL_READER, "official-reader",
                              plan->tensor_count, official->tensor_count, err, YVEX_ERR_FORMAT,
                              "pinned official-reader evidence is absent or mismatched");
    if (strlen(roundtrip->artifact_identity) != 64u ||
        !yvex_sha256_hex_is_valid(plan->payload_plan_identity) ||
        !yvex_sha256_hex_is_valid(emission->payload_byte_identity) ||
        !yvex_sha256_hex_is_valid(roundtrip->payload_byte_identity) ||
        strcmp(emission->payload_byte_identity, roundtrip->payload_byte_identity) != 0 ||
        !emission->execution_identity[0] ||
        (plan->required_execution_identity[0] &&
         strcmp(emission->execution_identity, plan->required_execution_identity) != 0))
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH, "identity-chain",
                              1u, 0u, err, YVEX_ERR_FORMAT,
                              "artifact or required quant execution identity is mismatched");
    if (!roundtrip->tokenizer_complete || !plan->tokenizer_token_count ||
        !plan->tokenizer_merge_count)
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_TOKENIZER_INCOMPLETE, "tokenizer",
                              1u, 0u, err, YVEX_ERR_FORMAT,
                              "complete tokenizer material was not admitted");
    if (roundtrip->tensor_count != plan->tensor_count ||
        roundtrip->payload_bytes_verified != plan->tensor_payload_bytes)
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE, "tensor-coverage",
                              plan->tensor_count, roundtrip->tensor_count, err, YVEX_ERR_FORMAT,
                              "complete tensor coverage is not bound to the artifact");

    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = request->artifact_path;
    artifact_options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &artifact_options, err);
    if (rc != YVEX_OK)
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_FILE_OPEN, "artifact-path", 1u, 0u,
                              err, (yvex_status)rc, "published artifact cannot be reopened");
    rc = yvex_artifact_snapshot_get(artifact, &snapshot, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK || !admission_snapshot_matches(&snapshot, emission)) {
        yvex_artifact_close(artifact);
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_FILE_DRIFT, "file-snapshot",
                              emission->file_size, rc == YVEX_OK ? snapshot.size : 0u, err,
                              YVEX_ERR_STATE,
                              "published artifact snapshot differs from validated emission");
    }
    yvex_artifact_close(artifact);

    out->artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX;
    out->metadata_count = plan->metadata_count;
    out->tensor_count = plan->tensor_count;
    out->payload_bytes = plan->tensor_payload_bytes;
    out->file_bytes = plan->final_file_bytes;
    out->source_snapshot_identity = plan->source_snapshot_identity;
    out->mapping_identity = plan->mapping_identity;
    out->file_snapshot = snapshot;
    yvex_core_text_copy(out->artifact_path, sizeof(out->artifact_path), request->artifact_path);
    artifact_identity_copy(out->payload_identity, plan->payload_identity);
    artifact_identity_copy(out->transform_identity, plan->transform_identity);
    artifact_identity_copy(out->profile_identity, plan->profile_identity);
    yvex_core_text_copy(out->profile_name, sizeof(out->profile_name), plan->profile_name);
    artifact_identity_copy(out->quant_execution_identity, emission->execution_identity);
    artifact_identity_copy(out->payload_plan_identity, plan->payload_plan_identity);
    artifact_identity_copy(out->payload_byte_identity, emission->payload_byte_identity);
    artifact_identity_copy(out->writer_plan_identity, plan->writer_plan_identity);
    artifact_identity_copy(out->artifact_identity, roundtrip->artifact_identity);
    yvex_core_text_copy(out->official_reader_revision, sizeof(out->official_reader_revision),
                       official->revision);
    out->tokenizer_complete = 1;
    out->native_reader_accepted = 1;
    out->official_reader_accepted = 1;
    out->payload_integrity_accepted = 1;
    out->materialization_input_ready = 1;
    out->runtime_supported = 0;
    out->artifact_bytes_hashed = roundtrip->bytes_hashed;
    out->artifact_identity_verified = 1;
    if (!admission_identity_encode(out, out->admission_identity))
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
                              "admission-identity", 1u, 0u, err, YVEX_ERR_BOUNDS,
                              "artifact admission identity encoding failed");
    out->complete = 1;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

static int artifact_admit_complete_catalog(
    const yvex_artifact *artifact,
    const yvex_complete_artifact_admission *catalog,
    yvex_complete_artifact_admission *out,
    yvex_artifact_admission_failure *failure, yvex_error *err) {
    yvex_artifact_snapshot snapshot;
    int rc;

    if (out)
        memset(out, 0, sizeof(*out));
    if (!artifact || !catalog || !out)
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT, "catalog", 1u, 0u,
            err, YVEX_ERR_INVALID_ARG,
            "opened artifact, complete catalog row, and output are required");
    if (!admission_complete_catalog_valid(catalog))
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
                              "pinned-identities", 1u, 0u, err, YVEX_ERR_FORMAT,
                              "complete artifact catalog identities are invalid");
    if (yvex_artifact_size(artifact) != catalog->file_bytes)
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH, "file-bytes",
            catalog->file_bytes, yvex_artifact_size(artifact), err, YVEX_ERR_FORMAT,
            "artifact extent differs from its complete catalog row");
    rc = yvex_artifact_snapshot_get(artifact, &snapshot, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK)
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_FILE_DRIFT, "file-snapshot",
                              catalog->file_bytes, 0u, err, (yvex_status)rc,
                              "complete catalog artifact snapshot is not stable");

    *out = *catalog;
    out->artifact_bytes_hashed = 0u;
    out->artifact_identity_verified = 0;
    out->file_snapshot = snapshot;
    yvex_core_text_copy(out->artifact_path, sizeof(out->artifact_path), yvex_artifact_path(artifact));
    if (!admission_identity_encode(out, out->admission_identity))
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
                              "admission-identity", 1u, 0u, err, YVEX_ERR_BOUNDS,
                              "pinned artifact admission identity encoding failed");
    out->complete = 1;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

static int artifact_admit_component_catalog(
    const yvex_artifact *artifact, const yvex_complete_artifact_admission *catalog,
    yvex_complete_artifact_admission *out, yvex_artifact_admission_failure *failure,
    yvex_error *err)
{
    yvex_artifact_snapshot snapshot;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    if (!artifact || !out || !admission_component_catalog_valid(catalog))
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT, "component-catalog", 1u,
            0u, err, YVEX_ERR_INVALID_ARG,
            "exact admitted component catalog row and opened artifact are required");
    if (yvex_artifact_size(artifact) != catalog->file_bytes)
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH, "file-bytes",
            catalog->file_bytes, yvex_artifact_size(artifact), err, YVEX_ERR_FORMAT,
            "component artifact extent differs from its admitted catalog row");
    rc = yvex_artifact_snapshot_get(artifact, &snapshot, err);
    if (rc == YVEX_OK) rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK)
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_FILE_DRIFT, "file-snapshot",
            catalog->file_bytes, 0u, err, (yvex_status)rc,
            "component artifact snapshot is not stable");

    *out = *catalog;
    out->file_snapshot = snapshot;
    yvex_core_text_copy(out->artifact_path, sizeof(out->artifact_path),
                        yvex_artifact_path(artifact));
    out->artifact_bytes_hashed = 0u;
    out->artifact_identity_verified = 0;
    out->complete = 1;
    if (!admission_identity_encode(out, out->admission_identity))
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
            "admission-identity", 1u, 0u, err, YVEX_ERR_BOUNDS,
            "component admission identity encoding failed");
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Reconcile a family catalog with the complete structural and storage inventory before trust. */
int yvex_artifact_admit_catalog(
    const yvex_artifact *artifact, const yvex_gguf *gguf, const yvex_tensor_table *tensors,
    const yvex_artifact_catalog_contract *contract, yvex_complete_artifact_admission *out,
    yvex_artifact_admission_failure *failure, yvex_error *err)
{
    unsigned long long qtype_counts[YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u] = {0};
    unsigned long long elements = 0ull, payload = 0ull, index;
    const yvex_complete_artifact_admission *catalog = contract ? contract->catalog : NULL;
    int rc;

    if (out) memset(out, 0, sizeof(*out));
    if (contract && contract->catalog &&
        contract->catalog->artifact_class == YVEX_ARTIFACT_CLASS_COMPLETE_YVEX) {
        if (gguf || tensors || contract->metadata || contract->storage ||
            contract->metadata_count || contract->storage_count ||
            contract->elements || contract->alignment)
            return admission_fail(
                failure, YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT,
                "complete-catalog-contract", 1u, 0u, err, YVEX_ERR_INVALID_ARG,
                "complete catalog admission does not accept component structure");
        return artifact_admit_complete_catalog(
            artifact, contract->catalog, out, failure, err);
    }
    if (!artifact || !gguf || !tensors || !contract || !catalog || !out ||
        !contract->metadata || !contract->metadata_count || !contract->storage ||
        !contract->storage_count || !contract->elements || !contract->alignment)
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT, "component-contract", 1u,
            0u, err, YVEX_ERR_INVALID_ARG,
            "exact component contract and structural artifact views are required");
    if (yvex_gguf_metadata_count(gguf) != catalog->metadata_count)
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH, "metadata-count",
            catalog->metadata_count, yvex_gguf_metadata_count(gguf), err,
            YVEX_ERR_FORMAT, "component metadata coverage differs from its catalog");
    {
        const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, "general.alignment");
        unsigned long long alignment = 0ull;

        if (!value || yvex_gguf_value_as_u64(value, &alignment) != YVEX_OK ||
            alignment != contract->alignment)
            return admission_fail(
                failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH, "general.alignment",
                contract->alignment, alignment, err, YVEX_ERR_FORMAT,
                "component alignment metadata differs from its catalog");
    }
    for (index = 0ull; index < contract->metadata_count; ++index) {
        const yvex_artifact_component_metadata *fact = &contract->metadata[index];
        const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, fact->key);
        const char *text = NULL;
        unsigned long long length = 0ull;
        size_t expected = fact->value ? strlen(fact->value) : 0u;

        if (!fact->key || !fact->value || !value ||
            yvex_gguf_value_as_string(value, &text, &length) != YVEX_OK ||
            length != expected || memcmp(text, fact->value, expected) != 0)
            return admission_fail(
                failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
                fact->key ? fact->key : "metadata", expected, length, err,
                YVEX_ERR_FORMAT, "component string metadata differs from its catalog");
    }
    if (yvex_tensor_table_count(tensors) != catalog->tensor_count)
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE, "tensor-count",
            catalog->tensor_count, yvex_tensor_table_count(tensors), err,
            YVEX_ERR_FORMAT, "component tensor coverage differs from its catalog");
    for (index = 0ull; index < contract->storage_count; ++index) {
        unsigned long long prior;

        if (contract->storage[index].qtype > YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID ||
            !contract->storage[index].tensors)
            return admission_fail(
                failure, YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT, "storage-contract",
                1u, 0u, err, YVEX_ERR_INVALID_ARG,
                "component storage contract contains an invalid population");
        for (prior = 0ull; prior < index; ++prior)
            if (contract->storage[prior].qtype == contract->storage[index].qtype)
                return admission_fail(
                    failure, YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT, "storage-contract",
                    1u, 0u, err, YVEX_ERR_INVALID_ARG,
                    "component storage contract repeats one qtype");
    }
    for (index = 0ull; index < yvex_tensor_table_count(tensors); ++index) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, index);
        yvex_tensor_shape_accounting accounting;

        rc = yvex_tensor_shape_accounting_validate(tensor, &accounting, err);
        if (rc != YVEX_OK)
            return admission_fail(
                failure, YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE, "tensor-shape",
                1u, index, err, (yvex_status)rc,
                "component tensor shape or storage accounting is invalid");
        if (tensor->ggml_type > YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID)
            return admission_fail(
                failure, YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE, "tensor-qtype",
                YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID, tensor->ggml_type, err,
                YVEX_ERR_UNSUPPORTED, "component tensor qtype is outside the admitted ABI");
        qtype_counts[tensor->ggml_type]++;
        if (!yvex_core_u64_add(elements, accounting.element_count, &elements) ||
            !yvex_core_u64_add(payload, accounting.storage_byte_count, &payload))
            return admission_fail(
                failure, YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE, "tensor-population",
                1u, index, err, YVEX_ERR_BOUNDS,
                "component aggregate tensor accounting overflowed");
    }
    for (index = 0ull; index <= YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID; ++index) {
        unsigned long long expected = 0ull, storage_index;

        for (storage_index = 0ull; storage_index < contract->storage_count; ++storage_index)
            if (contract->storage[storage_index].qtype == index)
                expected = contract->storage[storage_index].tensors;
        if (qtype_counts[index] != expected)
            return admission_fail(
                failure, YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE, "tensor-qtype-count",
                expected, qtype_counts[index], err, YVEX_ERR_FORMAT,
                "component qtype population differs from its catalog");
    }
    if (elements != contract->elements || payload != catalog->payload_bytes)
        return admission_fail(
            failure, YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE,
            elements != contract->elements ? "element-count" : "payload-bytes",
            elements != contract->elements ? contract->elements : catalog->payload_bytes,
            elements != contract->elements ? elements : payload, err, YVEX_ERR_FORMAT,
            "component aggregate storage differs from its catalog");
    rc = artifact_admit_component_catalog(artifact, catalog, out, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(
            artifact, out, NULL, NULL, failure, err);
    return rc;
}

/* Bind one opened artifact snapshot to its admitted exact-file identity. */
int yvex_artifact_admission_identity_verify(
    const yvex_artifact *artifact, yvex_complete_artifact_admission *admission,
    int (*progress)(void *context, unsigned long long completed,
                    unsigned long long total), void *progress_context,
    yvex_artifact_admission_failure *failure, yvex_error *err)
{
    yvex_complete_artifact_admission next;
    yvex_artifact_file_identity identity;
    yvex_artifact_snapshot snapshot;
    int rc;

    if (!artifact || !admission || !admission->complete ||
        !yvex_sha256_hex_is_valid(admission->artifact_identity))
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT,
                              "artifact-identity", 1u, 0u, err, YVEX_ERR_INVALID_ARG,
                              "complete artifact admission and opened handle are required");
    next = *admission;
    memset(&identity, 0, sizeof(identity));
    rc = yvex_artifact_identity_read_open_progress(
        artifact, &identity, progress, progress_context, err);
    if (rc == YVEX_OK)
        rc = yvex_artifact_snapshot_get(artifact, &snapshot, err);
    if (rc != YVEX_OK || identity.file_size != admission->file_bytes ||
        strcmp(identity.sha256, admission->artifact_identity) != 0 ||
        !yvex_artifact_snapshot_equal(&snapshot, &admission->file_snapshot))
        return admission_fail(failure, YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
                              "artifact-byte-identity", admission->file_bytes,
                              rc == YVEX_OK ? identity.file_size : 0u, err,
                              rc == YVEX_OK ? YVEX_ERR_FORMAT : (yvex_status)rc,
                              "opened artifact bytes do not match admitted identity");
    next.artifact_bytes_hashed = identity.file_size;
    next.artifact_identity_verified = 1;
    *admission = next;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

typedef struct {
    yvex_artifact_physical_compatibility *result;
    yvex_artifact_compatibility_failure *failure;
    yvex_error *error;
    const char *tensor_name;
    unsigned long long tensor_index;
    unsigned int dimension;
} compatibility_check;

_Static_assert(YVEX_SHA256_HEX_CAP == YVEX_GGUF_WRITER_IDENTITY_CAP,
               "compatibility identities share canonical SHA-256 capacity");

static const char compatibility_message[] =
    "writer plan and admitted artifact physical consequences disagree";

static int compatibility_fail(compatibility_check *check, yvex_artifact_compatibility_code code,
                              const char *field, unsigned long long expected,
                              unsigned long long actual, yvex_status status) {
    yvex_artifact_physical_compatibility *result = check->result;
    yvex_artifact_compatibility_failure *failure = check->failure;

    if (result) {
        result->physical_payload_compatible = 0;
        result->payload_digest_equal = 0;
        result->artifact_rebuild_required = 1;
        result->materialization_rebuild_required = 1;
    }
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->tensor_index = check->tensor_index;
        failure->dimension = check->dimension;
        failure->expected = expected;
        failure->actual = actual;
        yvex_core_text_copy(failure->field, sizeof(failure->field), field);
        if (check->tensor_name)
            yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), check->tensor_name);
    }
    yvex_error_set(check->error, status, "artifact.physical.compatibility", compatibility_message);
    return status;
}

static int compatibility_provenance_validate(const yvex_gguf_writer_plan_summary *plan,
                                             const yvex_complete_artifact_admission *admission,
                                             const yvex_artifact *artifact,
                                             compatibility_check *check) {
    yvex_artifact_snapshot snapshot;
    const char *identities[] = {
        plan->writer_plan_identity,      plan->transform_identity,
        plan->profile_identity,          plan->payload_plan_identity,
        admission->writer_plan_identity, admission->artifact_identity,
        admission->payload_identity,     admission->transform_identity,
        admission->profile_identity,     admission->quant_execution_identity,
        admission->payload_plan_identity, admission->payload_byte_identity,
        admission->admission_identity,
    };
    char admission_identity[YVEX_SHA256_HEX_CAP];
    size_t index;
    int rc;

    if (!plan || !plan->complete || plan->payload_bytes_read != 0u)
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_WRITER_PLAN, "writer-plan", 1u,
                                  plan && plan->complete ? plan->payload_bytes_read : 0u,
                                  YVEX_ERR_STATE);
    if (!admission->complete || admission->artifact_class != YVEX_ARTIFACT_CLASS_COMPLETE_YVEX ||
        !admission->materialization_input_ready || !admission->native_reader_accepted ||
        !admission->official_reader_accepted || !admission->payload_integrity_accepted ||
        !admission->artifact_identity_verified ||
        admission->artifact_bytes_hashed != admission->file_bytes)
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_ADMISSION,
                                  "complete-admission", 1u, 0u, YVEX_ERR_STATE);
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_hex_is_valid(identities[index]))
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_IDENTITY,
                                      "identity-encoding", 1u, 0u, YVEX_ERR_FORMAT);
    if (!admission_identity_encode(admission, admission_identity) ||
        strcmp(admission_identity, admission->admission_identity) != 0)
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_IDENTITY,
                                  "admission-record", 1u, 0u, YVEX_ERR_FORMAT);
    if (plan->source_snapshot_identity == 0u || plan->mapping_identity == 0u ||
        plan->source_snapshot_identity != admission->source_snapshot_identity ||
        plan->mapping_identity != admission->mapping_identity ||
        strcmp(plan->payload_identity, admission->payload_identity) != 0 ||
        strcmp(plan->profile_name, admission->profile_name) != 0 ||
        strcmp(plan->payload_plan_identity, admission->payload_plan_identity) != 0)
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_IDENTITY,
                                  "physical-provenance", 1u, 0u, YVEX_ERR_FORMAT);
    rc = yvex_artifact_snapshot_get(artifact, &snapshot, check->error);
    if (rc == YVEX_OK)
        rc = yvex_artifact_snapshot_validate(artifact, NULL, check->error);
    if (rc != YVEX_OK || !yvex_artifact_snapshot_equal(&snapshot, &admission->file_snapshot))
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_SNAPSHOT, "artifact-snapshot",
                                  1u, 0u, rc == YVEX_OK ? YVEX_ERR_STATE : (yvex_status)rc);
    return YVEX_OK;
}

static int compatibility_layout_validate(const yvex_gguf_writer_plan_summary *plan,
                                         const yvex_complete_artifact_admission *admission,
                                         const yvex_artifact *artifact, const yvex_gguf *gguf,
                                         compatibility_check *check) {
    const yvex_gguf_header *header = yvex_gguf_header_view(gguf);
    yvex_gguf_layout_result layout;
    unsigned long long planned_data_offset;
    unsigned long long planned_padding;
    int rc;

    rc = yvex_gguf_layout_validate(artifact, gguf, &layout, check->error);
    check->result->payload_bytes_read = layout.tensor_payload_bytes_read;
    if (rc != YVEX_OK || !layout.accepted || layout.tensor_payload_bytes_read != 0u) {
        check->tensor_name = layout.tensor_name;
        check->tensor_index = layout.tensor_index;
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_LAYOUT, "canonical-layout",
                                  YVEX_GGUF_LAYOUT_OK, layout.code,
                                  rc == YVEX_OK ? YVEX_ERR_FORMAT : (yvex_status)rc);
    }
    if (!yvex_core_u64_add(plan->structural_bytes, plan->pre_data_padding_bytes,
                           &planned_data_offset) ||
        !yvex_core_u64_add(layout.inter_tensor_padding_bytes, layout.final_tensor_padding_bytes,
                           &planned_padding))
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_LAYOUT, "layout-arithmetic",
                                  1u, 0u, YVEX_ERR_BOUNDS);
    if (!header || header->version != plan->gguf_version ||
        yvex_gguf_metadata_count(gguf) != plan->metadata_count ||
        admission->metadata_count != plan->metadata_count ||
        admission->tensor_count != plan->tensor_count || layout.alignment != plan->alignment ||
        layout.tensor_data_offset != planned_data_offset ||
        layout.raw_tensor_bytes != plan->tensor_payload_bytes ||
        planned_padding != plan->tensor_padding_bytes ||
        layout.data_section_span != plan->data_section_bytes ||
        layout.required_file_end != plan->final_file_bytes ||
        admission->payload_bytes != plan->tensor_payload_bytes ||
        admission->file_bytes != plan->final_file_bytes ||
        yvex_artifact_size(artifact) != plan->final_file_bytes)
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_LAYOUT, "aggregate-layout",
                                  plan->final_file_bytes, yvex_artifact_size(artifact),
                                  YVEX_ERR_FORMAT);
    return YVEX_OK;
}

static int compatibility_tensors_validate(const yvex_gguf_writer_plan *writer_plan,
                                          const yvex_gguf_writer_plan_summary *plan,
                                          const yvex_gguf *gguf, compatibility_check *check) {
    unsigned long long ordinal;

    if (plan->tensor_count != yvex_gguf_tensor_count(gguf))
        return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_COUNT, "tensor-count",
                                  plan->tensor_count, yvex_gguf_tensor_count(gguf),
                                  YVEX_ERR_FORMAT);
    for (ordinal = 0u; ordinal < plan->tensor_count; ++ordinal) {
        const yvex_gguf_writer_tensor *expected =
            yvex_gguf_writer_plan_tensor_at(writer_plan, ordinal);
        const yvex_gguf_tensor_info *actual = yvex_gguf_tensor_at(gguf, ordinal);
        unsigned long long raw_end;
        unsigned long long padded_end;
        unsigned long long padded_absolute;
        unsigned int dimension;
        yvex_gguf_layout_code interval;

        check->tensor_index = ordinal;
        check->dimension = UINT_MAX;
        check->tensor_name = expected ? expected->name : NULL;
        if (!expected || !actual)
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_RANGE, "tensor-row",
                                      1u, 0u, YVEX_ERR_FORMAT);
        if (strcmp(expected->name, actual->name) != 0)
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_NAME, "tensor-name",
                                      strlen(expected->name), actual->name_len, YVEX_ERR_FORMAT);
        if (expected->rank != actual->rank)
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_RANK, "tensor-rank",
                                      expected->rank, actual->rank, YVEX_ERR_FORMAT);
        if (expected->qtype != actual->ggml_type)
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_QTYPE,
                                      "tensor-qtype", expected->qtype, actual->ggml_type,
                                      YVEX_ERR_FORMAT);
        if (expected->raw_bytes != actual->storage_bytes)
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_BYTES,
                                      "tensor-raw-bytes", expected->raw_bytes,
                                      actual->storage_bytes, YVEX_ERR_FORMAT);
        for (dimension = 0u; dimension < expected->rank; ++dimension) {
            if (expected->dims[dimension] == actual->dims[dimension])
                continue;
            check->dimension = dimension;
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_DIMENSION,
                                      "tensor-dimension", expected->dims[dimension],
                                      actual->dims[dimension], YVEX_ERR_FORMAT);
        }
        check->dimension = UINT_MAX;
        interval = yvex_gguf_layout_interval_measure(actual->relative_offset, actual->storage_bytes,
                                                     plan->alignment, &raw_end, &padded_end);
        if (expected->relative_offset != actual->relative_offset ||
            expected->absolute_offset != actual->absolute_offset)
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_OFFSET,
                                      "tensor-offset", expected->absolute_offset,
                                      actual->absolute_offset, YVEX_ERR_FORMAT);
        padded_absolute = 0u;
        if (interval != YVEX_GGUF_LAYOUT_OK ||
            yvex_gguf_layout_sum_checked(yvex_gguf_tensor_data_offset(gguf), padded_end,
                                         &padded_absolute) != YVEX_GGUF_LAYOUT_OK ||
            expected->absolute_end != actual->absolute_end_offset ||
            expected->padded_bytes != padded_end - actual->relative_offset ||
            expected->padded_end != padded_absolute || !actual->range_addressable ||
            padded_absolute > plan->final_file_bytes)
            return compatibility_fail(check, YVEX_ARTIFACT_COMPATIBILITY_TENSOR_RANGE,
                                      "tensor-range", expected->padded_end, padded_absolute,
                                      YVEX_ERR_FORMAT);
        check->result->tensors_compared++;
    }
    return YVEX_OK;
}

/*
 * Prove a new semantic writer plan preserves an admitted physical GGUF exactly.
 *
 * Validates structural padding and fills one immutable caller-owned compatibility fact.
 */
int yvex_artifact_physical_compatibility_validate(
    const yvex_gguf_writer_plan *writer_plan, const yvex_complete_artifact_admission *admission,
    const yvex_artifact *artifact, const yvex_gguf *gguf, yvex_artifact_physical_compatibility *out,
    yvex_artifact_compatibility_failure *failure, yvex_error *err) {
    const yvex_gguf_writer_plan_summary *plan;
    compatibility_check check;
    int rc;

    if (out)
        memset(out, 0, sizeof(*out));
    if (failure)
        memset(failure, 0, sizeof(*failure));
    check = (compatibility_check){out, failure, err, NULL, ULLONG_MAX, UINT_MAX};
    if (!writer_plan || !admission || !artifact || !gguf || !out)
        return compatibility_fail(&check, YVEX_ARTIFACT_COMPATIBILITY_INVALID_ARGUMENT, "arguments",
                                  1u, 0u, YVEX_ERR_INVALID_ARG);
    plan = yvex_gguf_writer_plan_summary_get(writer_plan);
    if (!plan)
        return compatibility_fail(&check, YVEX_ARTIFACT_COMPATIBILITY_WRITER_PLAN, "writer-plan",
                                  1u, 0u, YVEX_ERR_STATE);
    *out = (yvex_artifact_physical_compatibility){
        .schema_version = YVEX_ARTIFACT_PHYSICAL_COMPATIBILITY_SCHEMA_VERSION,
        .source_snapshot_identity = admission->source_snapshot_identity,
        .mapping_identity = admission->mapping_identity,
        .tensor_count = plan->tensor_count,
        .payload_bytes = plan->tensor_payload_bytes,
        .artifact_rebuild_required = 1,
        .materialization_rebuild_required = 1,
    };
    artifact_identity_copy(out->writer_plan_identity, plan->writer_plan_identity);
    artifact_identity_copy(out->admitted_writer_plan_identity, admission->writer_plan_identity);
    artifact_identity_copy(out->artifact_identity, admission->artifact_identity);
    artifact_identity_copy(out->payload_identity, admission->payload_identity);
    artifact_identity_copy(out->writer_transform_identity, plan->transform_identity);
    artifact_identity_copy(out->admitted_transform_identity, admission->transform_identity);
    artifact_identity_copy(out->writer_profile_identity, plan->profile_identity);
    artifact_identity_copy(out->admitted_profile_identity, admission->profile_identity);
    artifact_identity_copy(out->quant_execution_identity, admission->quant_execution_identity);
    artifact_identity_copy(out->payload_plan_identity, admission->payload_plan_identity);
    artifact_identity_copy(out->payload_byte_identity, admission->payload_byte_identity);
    rc = compatibility_provenance_validate(plan, admission, artifact, &check);
    if (rc != YVEX_OK)
        return rc;
    rc = compatibility_layout_validate(plan, admission, artifact, gguf, &check);
    if (rc != YVEX_OK)
        return rc;
    rc = compatibility_tensors_validate(writer_plan, plan, gguf, &check);
    if (rc != YVEX_OK)
        return rc;

    out->tensor_inventory_equal = 1;
    out->qtype_equal = 1;
    out->layout_equal = 1;
    out->offset_equal = 1;
    out->payload_digest_equal =
        yvex_sha256_hex_is_valid(out->payload_plan_identity) &&
        yvex_sha256_hex_is_valid(out->payload_byte_identity) &&
        strcmp(plan->payload_plan_identity, admission->payload_plan_identity) == 0 &&
        admission->artifact_identity_verified &&
        admission->artifact_bytes_hashed == admission->file_bytes;
    if (!out->payload_digest_equal)
        return compatibility_fail(&check, YVEX_ARTIFACT_COMPATIBILITY_IDENTITY,
                                  "payload-identity-evidence", 1u, 0u, YVEX_ERR_FORMAT);
    out->physical_payload_compatible = 1;
    out->artifact_rebuild_required = 0;
    out->materialization_rebuild_required = 0;
    if (yvex_artifact_physical_compatibility_mismatch(
            out, admission, plan->transform_identity))
        return compatibility_fail(&check, YVEX_ARTIFACT_COMPATIBILITY_IDENTITY,
                                  "sealed-compatibility", 1u, 0u, YVEX_ERR_FORMAT);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

const char *yvex_artifact_physical_compatibility_mismatch(
    const yvex_artifact_physical_compatibility *proof,
    const yvex_complete_artifact_admission *admission,
    const char *logical_transform_identity)
{
    const char *identities[11];
    size_t index;
    if (!proof || !admission || !logical_transform_identity) return "record";
    identities[0] = proof->writer_plan_identity;
    identities[1] = proof->admitted_writer_plan_identity;
    identities[2] = proof->artifact_identity;
    identities[3] = proof->payload_identity;
    identities[4] = proof->writer_transform_identity;
    identities[5] = proof->admitted_transform_identity;
    identities[6] = proof->writer_profile_identity;
    identities[7] = proof->admitted_profile_identity;
    identities[8] = proof->quant_execution_identity;
    identities[9] = proof->payload_plan_identity;
    identities[10] = proof->payload_byte_identity;
    if (proof->schema_version != YVEX_ARTIFACT_PHYSICAL_COMPATIBILITY_SCHEMA_VERSION)
        return "schema-version";
    for (index = 0u; index < sizeof(identities) / sizeof(identities[0]); ++index)
        if (!yvex_sha256_hex_is_valid(identities[index])) return "identity-encoding";
    if (proof->source_snapshot_identity != admission->source_snapshot_identity)
        return "source-snapshot-identity";
    if (proof->mapping_identity != admission->mapping_identity) return "mapping-identity";
    if (proof->tensor_count != admission->tensor_count ||
        proof->tensors_compared != admission->tensor_count) return "tensor-count";
    if (proof->payload_bytes != admission->payload_bytes || proof->payload_bytes_read)
        return "payload-bytes";
    if (strcmp(proof->admitted_writer_plan_identity, admission->writer_plan_identity) != 0)
        return "admitted-writer-plan-identity";
    if (strcmp(proof->artifact_identity, admission->artifact_identity) != 0)
        return "artifact-identity";
    if (strcmp(proof->payload_identity, admission->payload_identity) != 0)
        return "payload-identity";
    if (strcmp(proof->writer_transform_identity, logical_transform_identity) != 0)
        return "writer-transform-identity";
    if (strcmp(proof->admitted_transform_identity, admission->transform_identity) != 0)
        return "admitted-transform-identity";
    if (strcmp(proof->admitted_profile_identity, admission->profile_identity) != 0)
        return "admitted-profile-identity";
    if (strcmp(proof->quant_execution_identity, admission->quant_execution_identity) != 0)
        return "quant-execution-identity";
    if (strcmp(proof->payload_plan_identity, admission->payload_plan_identity) != 0)
        return "payload-plan-identity";
    if (strcmp(proof->payload_byte_identity, admission->payload_byte_identity) != 0)
        return "payload-byte-identity";
    if (!proof->physical_payload_compatible || proof->artifact_rebuild_required ||
        proof->materialization_rebuild_required || !proof->tensor_inventory_equal ||
        !proof->qtype_equal || !proof->layout_equal || !proof->offset_equal ||
        !proof->payload_digest_equal) return "compatibility-verdict";
    return NULL;
}

const char *yvex_artifact_admission_code_name(yvex_artifact_admission_code code) {
    return artifact_name_at(artifact_admission_names,
                            sizeof(artifact_admission_names) / sizeof(artifact_admission_names[0]),
                            (unsigned int)code, "unknown");
}

static int roundtrip_fail(yvex_gguf_roundtrip_failure *failure, yvex_gguf_roundtrip_code code,
                          const char *name, unsigned long long metadata, unsigned long long tensor,
                          unsigned long long expected, unsigned long long actual,
                          unsigned long long offset, yvex_error *err, yvex_status status,
                          const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->metadata_index = metadata;
        failure->tensor_index = tensor;
        failure->expected = expected;
        failure->actual = actual;
        failure->file_offset = offset;
        if (name)
            yvex_core_text_copy(failure->name, sizeof(failure->name), name);
    }
    yvex_error_set(err, status, "gguf.roundtrip", message);
    return status;
}

static int roundtrip_array_count(const yvex_gguf *gguf, const char *key, unsigned int element_type,
                                 unsigned long long expected) {
    const yvex_gguf_value *value = yvex_gguf_metadata_find(gguf, key);
    yvex_gguf_array_info array;
    return value && yvex_gguf_value_array_info(value, &array) == YVEX_OK &&
           array.element_type == (yvex_gguf_value_type)element_type && array.count == expected;
}

/*
 * Hash one exact file range and optionally enforce zero-filled bytes.
 *
 * Admitted artifact, range, bounded scratch, hash states, and counters. Returns false on
 * arithmetic, read, hash, or zero-padding disagreement.
 */
static int roundtrip_hash_range(const yvex_artifact *artifact, unsigned long long offset,
                                unsigned long long byte_count, unsigned char *buffer,
                                size_t buffer_bytes, yvex_artifact_identity_stream *whole,
                                yvex_sha256 *terminal, int require_zero,
                                yvex_gguf_roundtrip_summary *summary,
                                unsigned long long *first_nonzero, yvex_error *err) {
    unsigned long long delivered = 0u;
    while (delivered < byte_count) {
        unsigned long long remaining = byte_count - delivered;
        size_t request = remaining < buffer_bytes ? (size_t)remaining : buffer_bytes;
        size_t index;
        if (offset > ULLONG_MAX - delivered ||
            yvex_artifact_read_at(artifact, offset + delivered, buffer, request, err) != YVEX_OK)
            return 0;
        if (yvex_artifact_identity_stream_update(whole, buffer, request, err) != YVEX_OK ||
            (terminal && !yvex_sha256_update(terminal, buffer, request)))
            return 0;
        if (require_zero)
            for (index = 0u; index < request; ++index)
                if (buffer[index] != 0u) {
                    if (first_nonzero)
                        *first_nonzero = offset + delivered + index;
                    return 0;
                }
        delivered += request;
        summary->bytes_hashed += request;
        summary->read_calls++;
    }
    return 1;
}

void yvex_gguf_roundtrip_options_default(yvex_gguf_roundtrip_options *options) {
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->verification_chunk_bytes = ROUNDTRIP_DEFAULT_CHUNK;
}

static void roundtrip_progress_publish(const yvex_gguf_roundtrip_options *options,
                                       const yvex_gguf_roundtrip_summary *summary,
                                       unsigned long long planned_file_bytes) {
    if (options && options->progress)
        options->progress(options->progress_context, summary, planned_file_bytes);
}

typedef struct {
    const char *path;
    const yvex_gguf_writer_plan *writer_plan;
    yvex_quant_digest_sink *digest_sink;
    const yvex_gguf_writer_plan_summary *plan;
    yvex_gguf_roundtrip_options options;
    yvex_gguf_roundtrip_summary *summary;
    yvex_gguf_roundtrip_failure *failure;
    yvex_error *err;
    yvex_artifact *artifact;
    yvex_artifact_snapshot before;
    yvex_artifact_snapshot after;
    yvex_gguf *gguf;
    unsigned char *buffer;
    const unsigned char *prefix;
    size_t prefix_bytes;
    yvex_artifact_identity_stream whole_hash;
    yvex_sha256 payload_hash;
    unsigned long long cursor;
    unsigned long long first_nonzero;
} roundtrip_context;

static int roundtrip_context_admit(roundtrip_context *context,
                                   const yvex_gguf_roundtrip_options *options) {
    if (!context->path || !context->path[0] || !context->plan || !context->digest_sink ||
        !context->summary)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_INVALID_ARGUMENT, NULL,
                              ULLONG_MAX, ULLONG_MAX, 1u, 0u, 0u, context->err,
                              YVEX_ERR_INVALID_ARG,
                              "artifact path, sealed plan, digest sink, and output are required");
    yvex_gguf_roundtrip_options_default(&context->options);
    if (options)
        context->options = *options;
    if (!context->options.verification_chunk_bytes ||
        context->options.verification_chunk_bytes > (size_t)SSIZE_MAX)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_INVALID_ARGUMENT, NULL,
                              ULLONG_MAX, ULLONG_MAX, 1u, context->options.verification_chunk_bytes,
                              0u, context->err, YVEX_ERR_INVALID_ARG,
                              "verification chunk size is invalid");
    return YVEX_OK;
}

static int roundtrip_open_container(roundtrip_context *context) {
    yvex_artifact_options artifact_options;
    yvex_gguf_reader_options reader_options;
    yvex_gguf_parse_result parse;
    yvex_gguf_layout_result layout;

    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = context->path;
    artifact_options.readonly = 1;
    if (yvex_artifact_open(&context->artifact, &artifact_options, context->err) != YVEX_OK ||
        yvex_artifact_snapshot_get(context->artifact, &context->before, context->err) != YVEX_OK)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_ARTIFACT_OPEN, context->path,
                              ULLONG_MAX, ULLONG_MAX, context->plan->final_file_bytes,
                              context->artifact ? yvex_artifact_size(context->artifact) : 0u, 0u,
                              context->err, YVEX_ERR_IO,
                              "artifact open or snapshot capture failed");
    if (context->before.size != context->plan->final_file_bytes)
        return roundtrip_fail(
            context->failure, YVEX_GGUF_ROUNDTRIP_HEADER_DIVERGENCE, context->path, ULLONG_MAX,
            ULLONG_MAX, context->plan->final_file_bytes, context->before.size, 0u, context->err,
            YVEX_ERR_FORMAT, "artifact file size differs from plan");

    yvex_gguf_reader_options_default(&reader_options);
    reader_options.max_metadata_entries = context->plan->metadata_count + 16u;
    reader_options.max_tensor_entries = context->plan->tensor_count;
    memset(&parse, 0, sizeof(parse));
    if (yvex_gguf_open_ex(&context->gguf, context->artifact, &reader_options, &parse,
                          context->err) != YVEX_OK)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_READER_REFUSAL, context->path,
                              parse.record_index, ULLONG_MAX, YVEX_GGUF_PARSE_OK, parse.code,
                              parse.byte_offset, context->err, YVEX_ERR_FORMAT,
                              "native GGUF reader refused emitted artifact");
    context->summary->reader_accepted = 1;
    memset(&layout, 0, sizeof(layout));
    if (yvex_gguf_layout_validate(context->artifact, context->gguf, &layout, context->err) !=
            YVEX_OK ||
        !layout.accepted)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_LAYOUT_REFUSAL,
                              layout.tensor_name, ULLONG_MAX, layout.tensor_index,
                              YVEX_GGUF_LAYOUT_OK, layout.code, layout.failure_absolute_offset,
                              context->err, YVEX_ERR_FORMAT,
                              "native global layout validator refused emitted artifact");
    context->summary->layout_accepted = 1;
    return YVEX_OK;
}

static int roundtrip_validate_directory(roundtrip_context *context) {
    const yvex_gguf_header *header = yvex_gguf_header_view(context->gguf);
    unsigned long long ordinal;

    if (!header || header->version != context->plan->gguf_version ||
        header->metadata_count != context->plan->metadata_count ||
        header->tensor_count != context->plan->tensor_count ||
        yvex_gguf_alignment(context->gguf) != context->plan->alignment ||
        yvex_gguf_tensor_data_offset(context->gguf) !=
            context->plan->structural_bytes + context->plan->pre_data_padding_bytes ||
        yvex_gguf_file_size(context->gguf) != context->plan->final_file_bytes)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_HEADER_DIVERGENCE,
                              context->path, ULLONG_MAX, ULLONG_MAX,
                              context->plan->final_file_bytes, yvex_gguf_file_size(context->gguf),
                              0u, context->err, YVEX_ERR_FORMAT,
                              "reader header/alignment/data offset differs from writer plan");
    context->summary->metadata_count = header->metadata_count;
    context->summary->tensor_count = header->tensor_count;
    if (context->plan->tokenizer_token_count &&
        (!roundtrip_array_count(context->gguf, "tokenizer.ggml.tokens", YVEX_GGUF_VALUE_STRING,
                                context->plan->tokenizer_token_count) ||
         !roundtrip_array_count(context->gguf, "tokenizer.ggml.token_type", YVEX_GGUF_VALUE_INT32,
                                context->plan->tokenizer_token_count) ||
         !roundtrip_array_count(context->gguf, "tokenizer.ggml.merges", YVEX_GGUF_VALUE_STRING,
                                context->plan->tokenizer_merge_count) ||
         !yvex_gguf_metadata_find(context->gguf, "tokenizer.huggingface.json") ||
         !yvex_gguf_metadata_find(context->gguf, "yvex.tokenizer.config.json")))
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_TOKENIZER_INCOMPLETE,
                              "tokenizer", ULLONG_MAX, ULLONG_MAX,
                              context->plan->tokenizer_token_count, 0u, 0u, context->err,
                              YVEX_ERR_FORMAT, "artifact tokenizer material is incomplete");
    context->summary->tokenizer_complete = context->plan->tokenizer_token_count != 0u;

    for (ordinal = 0u; ordinal < context->plan->tensor_count; ++ordinal) {
        const yvex_gguf_writer_tensor *expected =
            yvex_gguf_writer_plan_tensor_at(context->writer_plan, ordinal);
        const yvex_gguf_tensor_info *actual = yvex_gguf_tensor_at(context->gguf, ordinal);
        unsigned int dimension;
        int matches = expected && actual && actual->name_len == strlen(expected->name) &&
                      memcmp(actual->name, expected->name, actual->name_len) == 0 &&
                      actual->rank == expected->rank && actual->ggml_type == expected->qtype &&
                      actual->relative_offset == expected->relative_offset &&
                      actual->absolute_offset == expected->absolute_offset &&
                      actual->storage_bytes == expected->raw_bytes &&
                      actual->absolute_end_offset == expected->absolute_end;
        if (!matches)
            return roundtrip_fail(
                context->failure, YVEX_GGUF_ROUNDTRIP_TENSOR_DIVERGENCE,
                expected ? expected->name : NULL, ULLONG_MAX, ordinal,
                expected ? expected->raw_bytes : 0u, actual ? actual->storage_bytes : 0u,
                expected ? expected->absolute_offset : 0u, context->err, YVEX_ERR_FORMAT,
                "reader tensor directory row differs from writer plan");
        for (dimension = 0u; dimension < expected->rank; ++dimension)
            if (actual->dims[dimension] != expected->dims[dimension])
                return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_TENSOR_DIVERGENCE,
                                      expected->name, ULLONG_MAX, ordinal,
                                      expected->dims[dimension], actual->dims[dimension],
                                      expected->absolute_offset, context->err, YVEX_ERR_FORMAT,
                                      "reader tensor dimension differs from writer plan");
    }
    return YVEX_OK;
}

/*
 * Verify and hash the serialized structural prefix exactly once.
 *
 * Open artifact, sealed plan prefix, bounded chunk configuration. Allocates scratch and advances
 * whole-file identity and counters.
 */
static int roundtrip_verify_prefix(roundtrip_context *context) {
    context->prefix = yvex_gguf_writer_plan_prefix(context->writer_plan, &context->prefix_bytes);
    context->buffer = (unsigned char *)malloc(context->options.verification_chunk_bytes);
    if (!context->prefix || !context->buffer)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_ALLOCATION, NULL, ULLONG_MAX,
                              ULLONG_MAX, context->options.verification_chunk_bytes, 0u, 0u,
                              context->err, YVEX_ERR_NOMEM,
                              "roundtrip verification buffer allocation failed");
    context->summary->peak_owned_bytes = context->options.verification_chunk_bytes;
    yvex_artifact_identity_stream_init(&context->whole_hash);
    yvex_sha256_init(&context->payload_hash);
    if (!yvex_sha256_update_text(&context->payload_hash, "yvex.quant.payload.bytes.v1") ||
        !yvex_sha256_update_u64(&context->payload_hash, context->plan->tensor_count))
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_ARTIFACT_DIGEST, NULL,
                              ULLONG_MAX, ULLONG_MAX, context->plan->tensor_count, 0u, 0u,
                              context->err, YVEX_ERR_BOUNDS,
                              "payload identity initialization failed");
    while (context->cursor < context->prefix_bytes) {
        size_t remaining = context->prefix_bytes - (size_t)context->cursor;
        size_t request = remaining < context->options.verification_chunk_bytes
                             ? remaining
                             : context->options.verification_chunk_bytes;
        if (yvex_artifact_read_at(context->artifact, context->cursor, context->buffer, request,
                                  context->err) != YVEX_OK)
            return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_SHORT_READ, NULL,
                                  ULLONG_MAX, ULLONG_MAX, request, 0u, context->cursor,
                                  context->err, YVEX_ERR_IO, "structural prefix exact read failed");
        if (memcmp(context->buffer, context->prefix + context->cursor, request) != 0 ||
            yvex_artifact_identity_stream_update(&context->whole_hash, context->buffer, request,
                                                 context->err) != YVEX_OK)
            return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_PREFIX_DIVERGENCE, NULL,
                                  ULLONG_MAX, ULLONG_MAX, request, 0u, context->cursor,
                                  context->err, YVEX_ERR_FORMAT,
                                  "serialized structural prefix differs from writer plan");
        context->cursor += request;
        context->summary->bytes_hashed += request;
        context->summary->prefix_bytes_verified += request;
        context->summary->read_calls++;
    }
    roundtrip_progress_publish(&context->options, context->summary,
                               context->plan->final_file_bytes);
    return YVEX_OK;
}

/*
 * Verify one tensor payload digest and its canonical zero padding.
 *
 * Refuses missing digest, short read, digest mismatch, or dirty padding.
 */
static int roundtrip_verify_terminal(roundtrip_context *context, unsigned long long ordinal) {
    const yvex_gguf_writer_tensor *tensor =
        yvex_gguf_writer_plan_tensor_at(context->writer_plan, ordinal);
    yvex_quant_terminal_digest expected_digest;
    yvex_quant_failure quant_failure;
    yvex_sha256 terminal_hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char terminal_hex[YVEX_SHA256_HEX_BYTES];
    unsigned long long padding;

    if (!tensor || context->cursor != tensor->absolute_offset ||
        yvex_quant_digest_sink_terminal_at(context->digest_sink, ordinal, &expected_digest,
                                           &quant_failure, context->err) != YVEX_OK)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_PAYLOAD_DIGEST,
                              tensor ? tensor->name : NULL, ULLONG_MAX, ordinal,
                              tensor ? tensor->absolute_offset : 0u, context->cursor,
                              context->cursor, context->err, YVEX_ERR_FORMAT,
                              "terminal digest or canonical payload position is unavailable");
    yvex_sha256_init(&terminal_hash);
    if (!roundtrip_hash_range(context->artifact, context->cursor, tensor->raw_bytes,
                              context->buffer, context->options.verification_chunk_bytes,
                              &context->whole_hash, &terminal_hash, 0, context->summary, NULL,
                              context->err) ||
        !yvex_sha256_final(&terminal_hash, digest))
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_SHORT_READ, tensor->name,
                              ULLONG_MAX, ordinal, tensor->raw_bytes, 0u, context->cursor,
                              context->err, YVEX_ERR_IO, "terminal payload exact read failed");
    yvex_sha256_hex(digest, terminal_hex);
    if (expected_digest.delivered_bytes != tensor->raw_bytes ||
        strcmp(expected_digest.sha256, terminal_hex) != 0)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_PAYLOAD_DIGEST, tensor->name,
                              ULLONG_MAX, ordinal, expected_digest.delivered_bytes,
                              tensor->raw_bytes, context->cursor, context->err, YVEX_ERR_FORMAT,
                              "terminal payload digest differs from quant execution");
    if (!yvex_sha256_update_u64(&context->payload_hash, ordinal) ||
        !yvex_sha256_update_u64(&context->payload_hash, tensor->qtype) ||
        !yvex_sha256_update_u64(&context->payload_hash, tensor->raw_bytes) ||
        !yvex_sha256_update_text(&context->payload_hash, terminal_hex))
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_PAYLOAD_DIGEST,
                              tensor->name, ULLONG_MAX, ordinal, tensor->raw_bytes, 0u,
                              context->cursor, context->err, YVEX_ERR_BOUNDS,
                              "aggregate payload identity encoding failed");
    context->cursor += tensor->raw_bytes;
    context->summary->payload_bytes_verified += tensor->raw_bytes;
    padding = tensor->padded_bytes - tensor->raw_bytes;
    context->first_nonzero = ULLONG_MAX;
    if (padding &&
        !roundtrip_hash_range(context->artifact, context->cursor, padding, context->buffer,
                              context->options.verification_chunk_bytes, &context->whole_hash, NULL,
                              1, context->summary, &context->first_nonzero, context->err))
        return roundtrip_fail(
            context->failure,
            context->first_nonzero != ULLONG_MAX ? YVEX_GGUF_ROUNDTRIP_NONZERO_PADDING
                                                 : YVEX_GGUF_ROUNDTRIP_SHORT_READ,
            tensor->name, ULLONG_MAX, ordinal, 0u, 1u,
            context->first_nonzero != ULLONG_MAX ? context->first_nonzero : context->cursor,
            context->err, context->first_nonzero != ULLONG_MAX ? YVEX_ERR_FORMAT : YVEX_ERR_IO,
            "tensor padding is nonzero or unreadable");
    context->cursor += padding;
    context->summary->padding_bytes_verified += padding;
    context->summary->terminals_verified++;
    roundtrip_progress_publish(&context->options, context->summary,
                               context->plan->final_file_bytes);
    return YVEX_OK;
}

/*
 * Seal whole-file identity and prove the open inode remained stable.
 *
 * Finalizes artifact identity and publishes immutable snapshot facts.
 */
static int roundtrip_finish(roundtrip_context *context) {
    unsigned char payload_digest[YVEX_SHA256_DIGEST_BYTES];

    if (context->cursor != context->plan->final_file_bytes ||
        context->summary->bytes_hashed != context->plan->final_file_bytes ||
        yvex_artifact_identity_stream_final(&context->whole_hash, context->plan->final_file_bytes,
                                            context->summary->artifact_identity,
                                            context->err) != YVEX_OK)
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_ARTIFACT_DIGEST, NULL,
                              ULLONG_MAX, ULLONG_MAX, context->plan->final_file_bytes,
                              context->cursor, context->cursor, context->err, YVEX_ERR_FORMAT,
                              "whole-file byte coverage or artifact digest failed");
    if (!yvex_sha256_final(&context->payload_hash, payload_digest))
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_PAYLOAD_DIGEST, NULL,
                              ULLONG_MAX, ULLONG_MAX, context->plan->tensor_payload_bytes, 0u,
                              context->cursor, context->err, YVEX_ERR_BOUNDS,
                              "aggregate payload identity finalization failed");
    yvex_sha256_hex(payload_digest, context->summary->payload_byte_identity);
    context->summary->file_bytes = context->plan->final_file_bytes;
    context->summary->payload_accepted = 1;
    if (yvex_artifact_snapshot_validate(context->artifact, &context->after, context->err) !=
            YVEX_OK ||
        !yvex_artifact_snapshot_equal(&context->before, &context->after))
        return roundtrip_fail(context->failure, YVEX_GGUF_ROUNDTRIP_FILE_DRIFT, context->path,
                              ULLONG_MAX, ULLONG_MAX, context->before.size, context->after.size, 0u,
                              context->err, YVEX_ERR_STATE,
                              "artifact file identity drifted during roundtrip");
    context->summary->snapshot_stable = 1;
    context->summary->file_device = context->before.device;
    context->summary->file_inode = context->before.inode;
    context->summary->file_mtime_seconds = context->before.mtime_seconds;
    context->summary->file_mtime_nanoseconds = context->before.mtime_nanoseconds;
    context->summary->file_ctime_seconds = context->before.ctime_seconds;
    context->summary->file_ctime_nanoseconds = context->before.ctime_nanoseconds;
    context->summary->complete = 1;
    return YVEX_OK;
}

/*
 * Validate every planned byte of one complete temporary/final artifact.
 *
 * Bounded full-file reads plus deterministic summary/failure publication.
 */
int yvex_gguf_roundtrip_validate(const char *path, const yvex_gguf_writer_plan *writer_plan,
                                 yvex_quant_digest_sink *digest_sink,
                                 const yvex_gguf_roundtrip_options *options,
                                 yvex_gguf_roundtrip_summary *out,
                                 yvex_gguf_roundtrip_failure *failure, yvex_error *err) {
    roundtrip_context context;
    unsigned long long ordinal;
    int rc;

    if (out)
        memset(out, 0, sizeof(*out));
    memset(&context, 0, sizeof(context));
    context.path = path;
    context.writer_plan = writer_plan;
    context.digest_sink = digest_sink;
    context.plan = yvex_gguf_writer_plan_summary_get(writer_plan);
    context.summary = out;
    context.failure = failure;
    context.err = err;
    context.first_nonzero = ULLONG_MAX;
    rc = roundtrip_context_admit(&context, options);
    if (rc == YVEX_OK)
        rc = roundtrip_open_container(&context);
    if (rc == YVEX_OK)
        rc = roundtrip_validate_directory(&context);
    if (rc == YVEX_OK)
        rc = roundtrip_verify_prefix(&context);
    for (ordinal = 0u; rc == YVEX_OK && ordinal < context.plan->tensor_count; ++ordinal)
        rc = roundtrip_verify_terminal(&context, ordinal);
    if (rc == YVEX_OK)
        rc = roundtrip_finish(&context);
    if (rc == YVEX_OK) {
        if (failure)
            memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    free(context.buffer);
    yvex_gguf_close(context.gguf);
    yvex_artifact_close(context.artifact);
    if (rc != YVEX_OK && out)
        memset(out, 0, sizeof(*out));
    return rc;
}
