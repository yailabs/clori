/*
 * Exercises artifact layer artifact opening and range checking against tiny checked-in fixtures.
 * No model downloads or real model files are required.
 */
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/artifact.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/deepseek_v4.h>

#include "tests/test.h"

static int test_artifact_symlink_refusal(void)
{
    char root[] = "/tmp/yvex-artifact-XXXXXX";
    char real_dir[YVEX_ARTIFACT_PATH_CAP], held_dir[YVEX_ARTIFACT_PATH_CAP];
    char real_path[YVEX_ARTIFACT_PATH_CAP], held_path[YVEX_ARTIFACT_PATH_CAP];
    char moved_path[YVEX_ARTIFACT_PATH_CAP];
    char final_link[YVEX_ARTIFACT_PATH_CAP], parent_link[YVEX_ARTIFACT_PATH_CAP];
    char linked_path[YVEX_ARTIFACT_PATH_CAP];
    yvex_artifact_options options;
    yvex_artifact *artifact = NULL;
    yvex_error err;
    int fd, rc;

    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "artifact symlink root created");
    YVEX_TEST_ASSERT(snprintf(real_dir, sizeof(real_dir), "%s/real", root) <
                             (int)sizeof(real_dir) &&
                         snprintf(held_dir, sizeof(held_dir), "%s/held", root) <
                             (int)sizeof(held_dir) &&
                         snprintf(real_path, sizeof(real_path), "%s/model.gguf", real_dir) <
                             (int)sizeof(real_path) &&
                         snprintf(held_path, sizeof(held_path), "%s/held.gguf", real_dir) <
                             (int)sizeof(held_path) &&
                         snprintf(moved_path, sizeof(moved_path), "%s/model.gguf", held_dir) <
                             (int)sizeof(moved_path) &&
                         snprintf(final_link, sizeof(final_link), "%s/final.gguf", root) <
                             (int)sizeof(final_link) &&
                         snprintf(parent_link, sizeof(parent_link), "%s/linked", root) <
                             (int)sizeof(parent_link) &&
                         snprintf(linked_path, sizeof(linked_path), "%s/model.gguf", parent_link) <
                             (int)sizeof(linked_path),
                     "artifact symlink paths fit");
    YVEX_TEST_ASSERT(mkdir(real_dir, 0700) == 0, "artifact real directory created");
    fd = open(real_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    YVEX_TEST_ASSERT(fd >= 0 && write(fd, "GGUF", 4u) == 4 && close(fd) == 0,
                     "artifact regular fixture created");
    YVEX_TEST_ASSERT(symlink("real/model.gguf", final_link) == 0 &&
                         symlink("real", parent_link) == 0,
                     "artifact symlink fixtures created");

    memset(&options, 0, sizeof(options));
    options.readonly = 1;
    options.path = final_link;
    rc = yvex_artifact_open(&artifact, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO && artifact == NULL, "final artifact symlink refused");
    options.path = linked_path;
    rc = yvex_artifact_open(&artifact, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO && artifact == NULL,
                     "intermediate artifact symlink refused");
    YVEX_TEST_ASSERT(unlink(final_link) == 0 && unlink(parent_link) == 0,
                     "opening symlink fixtures removed");

    options.path = real_path;
    YVEX_TEST_ASSERT(yvex_artifact_open(&artifact, &options, &err) == YVEX_OK,
                     "regular artifact snapshot opened");
    YVEX_TEST_ASSERT(yvex_artifact_snapshot_validate(artifact, NULL, &err) == YVEX_OK,
                     "regular artifact snapshot validates");
    YVEX_TEST_ASSERT(rename(real_path, held_path) == 0 && symlink("held.gguf", real_path) == 0,
                     "final snapshot path replaced by symlink");
    YVEX_TEST_ASSERT(yvex_artifact_snapshot_validate(artifact, NULL, &err) == YVEX_ERR_FORMAT,
                     "snapshot validation refuses final symlink");
    YVEX_TEST_ASSERT(unlink(real_path) == 0 && rename(held_path, real_path) == 0,
                     "final snapshot path restored");
    yvex_artifact_close(artifact);
    artifact = NULL;

    YVEX_TEST_ASSERT(yvex_artifact_open(&artifact, &options, &err) == YVEX_OK,
                     "artifact reopened for parent drift");
    YVEX_TEST_ASSERT(rename(real_dir, held_dir) == 0 && symlink("held", real_dir) == 0,
                     "snapshot parent replaced by symlink");
    YVEX_TEST_ASSERT(yvex_artifact_snapshot_validate(artifact, NULL, &err) == YVEX_ERR_FORMAT,
                     "snapshot validation refuses intermediate symlink");
    yvex_artifact_close(artifact);
    YVEX_TEST_ASSERT(unlink(real_dir) == 0 && unlink(moved_path) == 0 &&
                         rmdir(held_dir) == 0 && rmdir(root) == 0,
                     "artifact symlink fixtures cleaned narrowly");
    return 0;
}

typedef struct {
    const char *filename;
    unsigned long long file_bytes;
    unsigned long long payload_bytes;
    const char *profile_identity;
    const char *artifact_identity;
    const char *quant_execution_identity;
    const char *payload_plan_identity;
    const char *payload_byte_identity;
    const char *writer_plan_identity;
    const char *admission_identity;
} deepseek_catalog_fixture;

static int test_deepseek_catalog_entry(const char *root,
                                       const deepseek_catalog_fixture *fixture)
{
    char path[YVEX_ARTIFACT_PATH_CAP];
    yvex_artifact_options options = {0};
    yvex_artifact *artifact = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure;
    yvex_error err;
    int fd;

    YVEX_TEST_ASSERT(snprintf(path, sizeof(path), "%s/%s", root, fixture->filename) <
                         (int)sizeof(path),
                     "variant catalog path fits");
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    YVEX_TEST_ASSERT(fd >= 0 && ftruncate(fd, (off_t)fixture->file_bytes) == 0 &&
                         close(fd) == 0,
                     "variant sparse extent created");
    options.path = path;
    options.readonly = 1;
    YVEX_TEST_ASSERT(yvex_artifact_open(&artifact, &options, &err) == YVEX_OK,
                     "variant sparse artifact opened");
    YVEX_TEST_ASSERT(yvex_artifact_admit_deepseek(artifact, &admission, &failure, &err) == YVEX_OK,
                     "variant catalog admission reconstructed");
    YVEX_TEST_ASSERT(admission.file_bytes == fixture->file_bytes &&
                         admission.payload_bytes == fixture->payload_bytes &&
                         strcmp(admission.profile_identity, fixture->profile_identity) == 0 &&
                         strcmp(admission.artifact_identity, fixture->artifact_identity) == 0 &&
                         strcmp(admission.quant_execution_identity,
                                fixture->quant_execution_identity) == 0 &&
                         strcmp(admission.payload_plan_identity,
                                fixture->payload_plan_identity) == 0 &&
                         strcmp(admission.payload_byte_identity,
                                fixture->payload_byte_identity) == 0 &&
                         strcmp(admission.writer_plan_identity,
                                fixture->writer_plan_identity) == 0 &&
                         strcmp(admission.admission_identity, fixture->admission_identity) == 0,
                     "variant catalog binds exact admitted identities");
    YVEX_TEST_ASSERT(!admission.artifact_identity_verified && admission.artifact_bytes_hashed == 0u,
                     "catalog reconstruction does not fabricate byte verification");
    {
        yvex_complete_artifact_admission mismatched = admission;
        yvex_complete_artifact_admission rejected;
        yvex_artifact_catalog_contract contract = {0};

        mismatched.file_bytes++;
        contract.catalog = &mismatched;
        YVEX_TEST_ASSERT(
            yvex_artifact_admit_catalog(
                artifact, NULL, NULL, &contract, &rejected, &failure,
                &err) == YVEX_ERR_FORMAT &&
                failure.code == YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH &&
                strcmp(failure.field, "file-bytes") == 0 && !rejected.complete,
            "generic catalog admission refuses a mismatched physical extent");
        mismatched.file_bytes--;
        contract.catalog = &mismatched;
        contract.alignment = 32ull;
        YVEX_TEST_ASSERT(
            yvex_artifact_admit_catalog(
                artifact, NULL, NULL, &contract, &rejected, &failure,
                &err) == YVEX_ERR_INVALID_ARG &&
                failure.code == YVEX_ARTIFACT_ADMISSION_INVALID_ARGUMENT &&
                strcmp(failure.field, "complete-catalog-contract") == 0 &&
                !rejected.complete,
            "complete catalog admission refuses component structure");
    }
    yvex_artifact_close(artifact);
    YVEX_TEST_ASSERT(unlink(path) == 0, "variant sparse artifact cleaned");
    return 0;
}

static int test_deepseek_variant_admission_catalog(void)
{
    char root[] = "/tmp/yvex-artifact-variant-XXXXXX";
    const deepseek_catalog_fixture selected = {
        .filename = "selected.gguf",
        .file_bytes = YVEX_SELECTED_DEEPSEEK_FILE_BYTES,
        .payload_bytes = YVEX_SELECTED_DEEPSEEK_PAYLOAD_BYTES,
        .profile_identity = YVEX_SELECTED_DEEPSEEK_PROFILE_IDENTITY,
        .artifact_identity = YVEX_SELECTED_DEEPSEEK_ARTIFACT_IDENTITY,
        .quant_execution_identity = YVEX_SELECTED_DEEPSEEK_EXECUTION_IDENTITY,
        .payload_plan_identity = YVEX_SELECTED_DEEPSEEK_PAYLOAD_PLAN_IDENTITY,
        .payload_byte_identity = YVEX_SELECTED_DEEPSEEK_PAYLOAD_BYTE_IDENTITY,
        .writer_plan_identity = YVEX_SELECTED_DEEPSEEK_WRITER_PLAN_IDENTITY,
        .admission_identity =
            "d8966a5222ef10f612595c657cbcf0a9cf557e277cb28bb44d85ad89c3bf42a0",
    };
    const deepseek_catalog_fixture native_drafter = {
        .filename = "native-drafter.gguf",
        .file_bytes = 98018204640ull,
        .payload_bytes = 98006498296ull,
        .profile_identity =
            "6a99e9f7c374e3f718cce705002bf2b799db9cc1b86f65091631857f52c1c587",
        .artifact_identity =
            "59c4649b19bb9f3eb7c01559e12ae52c3d4fbd067957e35de0a1a851759c7cc1",
        .quant_execution_identity =
            "35002244d5854a2d51b877ea31614cd985c9795d11c7e0904ed3475fec7fcb77",
        .payload_plan_identity =
            "e83545c729b219d327d4a437d499b73407648c94748ba7fda13905baace15c3e",
        .payload_byte_identity =
            "c79712bb85e31ebdcbd71ef0256709a001ae4cc62c4150ba8726d5dc5722dcd0",
        .writer_plan_identity =
            "2d4694925c02c04811ea846f389a94dbf524d26809a292c93f2c46ca8f05a025",
        .admission_identity =
            "9a6f6844e47dd7214b4bf12dd14a1ec34f0e88bc85c68cb00bba59fb674df6d9",
    };

    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "variant catalog root created");
    YVEX_TEST_ASSERT(test_deepseek_catalog_entry(root, &selected) == 0,
                     "selected DeepSeek catalog entry is exact");
    YVEX_TEST_ASSERT(test_deepseek_catalog_entry(root, &native_drafter) == 0,
                     "native-drafter DeepSeek catalog entry is exact");
    YVEX_TEST_ASSERT(rmdir(root) == 0, "variant catalog root cleaned");
    return 0;
}

static int test_component_admission_catalog(void)
{
    yvex_artifact_component_metadata metadata[] = {
        {"general.architecture", "llama"}, {"general.name", "yvex-test"},
    };
    yvex_artifact_component_storage storage[] = {{YVEX_GGUF_QTYPE_F32, 1ull}};
    yvex_complete_artifact_admission catalog = {
        .artifact_class = YVEX_ARTIFACT_CLASS_COMPONENT_YVEX,
        .metadata_count = 5u,
        .tensor_count = 1u,
        .payload_bytes = 128u,
        .file_bytes = 416u,
        .source_snapshot_identity = 0x11223344u,
        .mapping_identity = 0x55667788u,
        .payload_identity =
            "1111111111111111111111111111111111111111111111111111111111111111",
        .transform_identity =
            "2222222222222222222222222222222222222222222222222222222222222222",
        .profile_identity =
            "3333333333333333333333333333333333333333333333333333333333333333",
        .profile_name = "fixture-component-f32-v1",
        .quant_execution_identity =
            "4444444444444444444444444444444444444444444444444444444444444444",
        .payload_plan_identity =
            "5555555555555555555555555555555555555555555555555555555555555555",
        .payload_byte_identity =
            "6666666666666666666666666666666666666666666666666666666666666666",
        .writer_plan_identity =
            "7777777777777777777777777777777777777777777777777777777777777777",
        .artifact_identity =
            "3ad71e86689ae7d85d471dd9879702449551c945cb5e0ae50f02edc1fd99af44",
        .official_reader_revision = YVEX_GGUF_OFFICIAL_READER_REVISION,
        .logical_target = "fixture-target",
        .logical_component = "audio_vae",
        .logical_component_identity =
            "9999999999999999999999999999999999999999999999999999999999999999",
        .native_reader_accepted = 1,
        .official_reader_accepted = 1,
        .payload_integrity_accepted = 1,
        .materialization_input_ready = 1,
    };
    yvex_artifact_catalog_contract contract = {
        &catalog, metadata, storage, 2ull, 1ull, 32ull, 32ull,
    };
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure;
    yvex_artifact_options options = {0};
    yvex_artifact *artifact = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_gguf *gguf = NULL;
    yvex_error err;

    options.path = "tests/fixtures/gguf/valid-metadata-tensors.gguf";
    options.readonly = 1;
    YVEX_TEST_ASSERT(yvex_artifact_open(&artifact, &options, &err) == YVEX_OK,
                     "component catalog artifact opened");
    YVEX_TEST_ASSERT(yvex_gguf_open(&gguf, artifact, &err) == YVEX_OK &&
                         yvex_tensor_table_from_gguf(&tensors, gguf, &err) == YVEX_OK,
                     "component catalog structural views opened");
    YVEX_TEST_ASSERT(yvex_artifact_admit_catalog(
                         artifact, gguf, tensors, &contract, &admission, &failure,
                         &err) == YVEX_OK &&
                         admission.complete &&
                         admission.artifact_class == YVEX_ARTIFACT_CLASS_COMPONENT_YVEX &&
                         strcmp(admission.logical_component, "audio_vae") == 0 &&
                         !admission.tokenizer_complete &&
                         admission.artifact_identity_verified &&
                         admission.artifact_bytes_hashed == catalog.file_bytes &&
                         yvex_sha256_hex_is_valid(admission.admission_identity),
                     "component contract reconciles structure before publishing trust");
    metadata[1].value = "wrong";
    YVEX_TEST_ASSERT(yvex_artifact_admit_catalog(
                         artifact, gguf, tensors, &contract, &admission, &failure,
                         &err) == YVEX_ERR_FORMAT &&
                         failure.code == YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
                     "component contract refuses metadata identity drift");
    metadata[1].value = "yvex-test";
    storage[0].tensors = 2ull;
    YVEX_TEST_ASSERT(yvex_artifact_admit_catalog(
                         artifact, gguf, tensors, &contract, &admission, &failure,
                         &err) == YVEX_ERR_FORMAT &&
                         failure.code == YVEX_ARTIFACT_ADMISSION_TENSOR_COVERAGE,
                     "component contract refuses qtype population drift");
    storage[0].tensors = 1ull;
    catalog.file_bytes++;
    YVEX_TEST_ASSERT(yvex_artifact_admit_catalog(
                         artifact, gguf, tensors, &contract, &admission, &failure,
                         &err) == YVEX_ERR_FORMAT &&
                         failure.code == YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH,
                     "component catalog refuses wrong artifact extent");
    catalog.file_bytes--;
    strcpy(catalog.artifact_identity,
           "8888888888888888888888888888888888888888888888888888888888888888");
    YVEX_TEST_ASSERT(yvex_artifact_admit_catalog(
                         artifact, gguf, tensors, &contract, &admission, &failure,
                         &err) == YVEX_ERR_FORMAT &&
                         failure.code == YVEX_ARTIFACT_ADMISSION_IDENTITY_MISMATCH &&
                         !admission.artifact_identity_verified,
                     "component catalog refuses same-size content identity drift");
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

int yvex_test_artifact(void)
{
    const char *fixture = "tests/fixtures/gguf/valid-minimal.gguf";
    yvex_artifact_options options;
    yvex_artifact *artifact;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(test_artifact_symlink_refusal() == 0,
                     "artifact symlink lifecycle refuses unsafe paths");
    YVEX_TEST_ASSERT(test_deepseek_variant_admission_catalog() == 0,
                     "DeepSeek physical catalog admits the exact candidate");
    YVEX_TEST_ASSERT(test_component_admission_catalog() == 0,
                     "component physical catalog admits exact component facts");

    memset(&options, 0, sizeof(options));
    options.path = fixture;
    options.readonly = 1;
    options.map = 1;

    artifact = NULL;
    rc = yvex_artifact_open(&artifact, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open valid artifact");
    YVEX_TEST_ASSERT(artifact != NULL, "artifact non-null");
    YVEX_TEST_ASSERT_STREQ(yvex_artifact_path(artifact), fixture, "artifact path");
    YVEX_TEST_ASSERT(yvex_artifact_size(artifact) == 32, "artifact size");
    YVEX_TEST_ASSERT(yvex_artifact_is_mapped(artifact) == 1, "artifact mapping explicit");
    YVEX_TEST_ASSERT(yvex_artifact_data(artifact) != NULL, "artifact data");
    yvex_artifact_close(artifact);

    artifact = NULL;
    options.map = 0;
    rc = yvex_artifact_open(&artifact, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open file-backed artifact");
    YVEX_TEST_ASSERT(yvex_artifact_is_mapped(artifact) == 0, "file-backed artifact not mapped");
    YVEX_TEST_ASSERT(yvex_artifact_data(artifact) == NULL, "unmapped artifact has no payload pointer");
    {
        unsigned char magic[4];
        rc = yvex_artifact_read_at(artifact, 0ull, magic, sizeof(magic), &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK, "positioned artifact read");
        YVEX_TEST_ASSERT(memcmp(magic, "GGUF", sizeof(magic)) == 0, "positioned bytes");
    }
    rc = yvex_artifact_cache_release(artifact, 0ull, yvex_artifact_size(artifact), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "verified artifact cache range released");
    rc = yvex_artifact_cache_release(artifact, yvex_artifact_size(artifact), 1ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "out-of-range cache release refused");
    yvex_artifact_close(artifact);
    rc = yvex_artifact_cache_release(NULL, 0ull, 0ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG, "missing artifact cache release refused");

    artifact = NULL;
    options.readonly = 0;
    rc = yvex_artifact_open(&artifact, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED, "writable artifact mode refused");
    YVEX_TEST_ASSERT(artifact == NULL, "writable refusal leaves artifact null");
    options.readonly = 1;

    artifact = NULL;
    options.path = "tests/fixtures/gguf/missing.gguf";
    rc = yvex_artifact_open(&artifact, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO, "missing file returns IO");
    YVEX_TEST_ASSERT(artifact == NULL, "missing artifact null");

    options.path = NULL;
    rc = yvex_artifact_open(&artifact, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG, "null path invalid arg");

    rc = yvex_artifact_open(NULL, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG, "null out invalid arg");

    rc = yvex_range_check(24, 0, 24, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "full range ok");
    rc = yvex_range_check(24, 24, 0, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "zero range at EOF ok");
    rc = yvex_range_check(24, 25, 0, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "offset out of bounds");
    rc = yvex_range_check(24, 23, 2, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "range out of bounds");
    rc = yvex_range_check(ULLONG_MAX, ULLONG_MAX, 1, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "overflow-like range rejected");

    yvex_artifact_close(NULL);
    return 0;
}
