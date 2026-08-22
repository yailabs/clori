/*
 * Exercises artifact layer artifact opening and range checking against tiny checked-in fixtures.
 * No model downloads or real model files are required.
 */
#include <errno.h>
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

static int copy_fixture(const char *source, const char *destination)
{
    unsigned char buffer[4096];
    int input = -1, output = -1, result = 0;
    ssize_t count;

    input = open(source, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    output = open(destination, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (input < 0 || output < 0) goto done;
    for (;;) {
        do {
            count = read(input, buffer, sizeof(buffer));
        } while (count < 0 && errno == EINTR);
        if (count <= 0) break;
        size_t offset = 0u;
        while (offset < (size_t)count) {
            ssize_t written;
            do {
                written = write(output, buffer + offset, (size_t)count - offset);
            } while (written < 0 && errno == EINTR);
            if (written <= 0) goto done;
            offset += (size_t)written;
        }
    }
    result = count == 0 && fsync(output) == 0;
done:
    if (input >= 0) (void)close(input);
    if (output >= 0 && close(output) != 0) result = 0;
    return result;
}

static int test_verified_reopen_lease(void)
{
    char root[] = "/tmp/yvex-artifact-reopen-XXXXXX";
    char artifact_path[YVEX_ARTIFACT_PATH_CAP], cache_root[YVEX_ARTIFACT_PATH_CAP];
    char artifact_dir[YVEX_ARTIFACT_PATH_CAP], lease_path[YVEX_ARTIFACT_PATH_CAP];
    char receipt_root[YVEX_ARTIFACT_PATH_CAP];
    yvex_artifact_options options = {0};
    yvex_artifact_file_identity identity;
    yvex_artifact_reopen_lease lease;
    yvex_artifact *artifact = NULL;
    yvex_error err;
    unsigned char corrupt = 0u;
    int artifact_count, cache_count, descriptor;

    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "verified-reopen root is isolated");
    artifact_count = snprintf(artifact_path, sizeof(artifact_path), "%s/model.gguf", root);
    cache_count = snprintf(cache_root, sizeof(cache_root), "%s/cache", root);
    YVEX_TEST_ASSERT(artifact_count > 0 &&
                         artifact_count < (int)sizeof(artifact_path) && cache_count > 0 &&
                         cache_count < (int)sizeof(cache_root) &&
                         copy_fixture("tests/fixtures/gguf/valid-minimal.gguf", artifact_path),
                     "verified-reopen fixture is isolated");
    options.path = artifact_path;
    options.readonly = 1;
    YVEX_TEST_ASSERT(yvex_artifact_open(&artifact, &options, &err) == YVEX_OK &&
                         yvex_artifact_identity_read_open(artifact, &identity, &err) == YVEX_OK,
                     "verified-reopen fixture owns one complete identity");
    YVEX_TEST_ASSERT(yvex_artifact_reopen_lease_check(
                         artifact, identity.sha256, cache_root, &lease, &err) == YVEX_OK &&
                         !lease.verified && !lease.receipt_present,
                     "missing verified-reopen lease requires a full hash");
    YVEX_TEST_ASSERT(yvex_artifact_reopen_lease_publish(
                         artifact, identity.sha256, cache_root, &lease, &err) == YVEX_OK &&
                         lease.verified && lease.receipt_valid,
                     "complete verification publishes one content-addressed lease");
    YVEX_TEST_ASSERT(snprintf(lease_path, sizeof(lease_path), "%s", lease.path) > 0 &&
                         strlen(lease.path) < sizeof(lease_path),
                     "verified-reopen lease path is bounded");
    YVEX_TEST_ASSERT(yvex_artifact_reopen_lease_check(
                         artifact, identity.sha256, cache_root, &lease, &err) == YVEX_OK &&
                         lease.verified && lease.receipt_present && lease.receipt_valid,
                     "unchanged artifact snapshot admits verified reopen");
    descriptor = open(lease.path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    YVEX_TEST_ASSERT(descriptor >= 0 && pwrite(descriptor, &corrupt, 1u, 0) == 1 &&
                         fsync(descriptor) == 0 && close(descriptor) == 0,
                     "verified-reopen corruption fixture is durable");
    YVEX_TEST_ASSERT(yvex_artifact_reopen_lease_check(
                         artifact, identity.sha256, cache_root, &lease, &err) == YVEX_OK &&
                         !lease.verified && lease.receipt_present && !lease.receipt_valid,
                     "corrupt verified-reopen lease falls back to a full hash");
    yvex_artifact_close(artifact);
    artifact = NULL;
    descriptor = open(artifact_path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    YVEX_TEST_ASSERT(descriptor >= 0 && pwrite(descriptor, "X", 1u, 0) == 1 &&
                         fsync(descriptor) == 0 && close(descriptor) == 0 &&
                         yvex_artifact_open(&artifact, &options, &err) == YVEX_OK &&
                         yvex_artifact_reopen_lease_check(
                             artifact, identity.sha256, cache_root, &lease, &err) == YVEX_OK &&
                         !lease.verified && !lease.receipt_present,
                     "artifact snapshot drift cannot reuse the prior lease");
    yvex_artifact_close(artifact);
    artifact_count = snprintf(artifact_dir, sizeof(artifact_dir),
                              "%s/artifact-reopen/%s", cache_root, identity.sha256);
    cache_count = snprintf(receipt_root, sizeof(receipt_root), "%s/artifact-reopen",
                           cache_root);
    YVEX_TEST_ASSERT(artifact_count > 0 && artifact_count < (int)sizeof(artifact_dir) &&
                         cache_count > 0 && cache_count < (int)sizeof(receipt_root) &&
                         unlink(lease_path) == 0 && unlink(artifact_path) == 0 &&
                         rmdir(artifact_dir) == 0 && rmdir(receipt_root) == 0 &&
                         rmdir(cache_root) == 0 && rmdir(root) == 0,
                     "verified-reopen fixture is cleaned narrowly");
    return 0;
}

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
        .payload_bytes = 108274154488ull,
        .profile_identity =
            "a48d43c8594999a1af3a5b1f572b34a5823042cb767832d558642bb804b036c5",
        .artifact_identity =
            "bf80bd7372e9ff754cd61d8f6e849ca8eff2177fad40840a2dad8e840b35690a",
        .quant_execution_identity =
            "777559149e4e8421c34299da78f63f6b0d296a91005d7670196164c3c72b62af",
        .payload_plan_identity =
            "8d1a89e794363c0aaf1c721b07c0661ea03f9680691d0113543b2540297b69e7",
        .payload_byte_identity =
            "6dce1edb82810715687d40c6d62273e992cfe9e0aa610cb9598447e06fb7099f",
        .writer_plan_identity =
            "1ba1ceaa709862145b1a145e938cf03327cd58da27bca42ade2f884e2b2fc635",
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
    const deepseek_catalog_fixture compact_mxfp4 = {
        .filename = "compact-mxfp4.gguf",
        .file_bytes = 95050210304ull,
        .payload_bytes = 95038503928ull,
        .profile_identity =
            "b9825a070028a66af28cdb25614f7a86c6ad1ec396eed6ae961039db1507ce0e",
        .artifact_identity =
            "d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53",
        .quant_execution_identity =
            "ca591438ac7296fa9b3d1ad74415508d57d92835ab783d01b7da9bfec561e8d7",
        .payload_plan_identity =
            "2cf8c2c41cb13ad228a2193e55ae56da210a7b06d356ec74ad79155a3e1cd1e0",
        .payload_byte_identity =
            "85b52eae100a482f557611ac9b5e84e9bd133525d01a3a7a27fc7520aa819fd5",
        .writer_plan_identity =
            "c4484184f0d4b3aeba9ae306b3247f4e3134e734ecfa6cd0f5d79ac24c524bce",
        .admission_identity =
            "fba17e4b1b50ba8a73ee7ab2c8c0ace1e021feb01a57aa91001c9b459b8ac161",
    };

    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "variant catalog root created");
    YVEX_TEST_ASSERT(test_deepseek_catalog_entry(root, &selected) == 0,
                     "selected DeepSeek catalog entry is exact");
    YVEX_TEST_ASSERT(test_deepseek_catalog_entry(root, &native_drafter) == 0,
                     "native-drafter DeepSeek catalog entry is exact");
    YVEX_TEST_ASSERT(test_deepseek_catalog_entry(root, &compact_mxfp4) == 0,
                     "compact MXFP4 DeepSeek catalog entry is exact");
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
    YVEX_TEST_ASSERT(test_verified_reopen_lease() == 0,
                     "verified reopen is exact, rebuildable, and fail closed");

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
