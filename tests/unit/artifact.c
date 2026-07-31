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

static int test_deepseek_variant_admission_catalog(void)
{
    char root[] = "/tmp/yvex-artifact-variant-XXXXXX";
    char path[YVEX_ARTIFACT_PATH_CAP];
    yvex_artifact_options options = {0};
    yvex_artifact *artifact = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure failure;
    yvex_error err;
    int fd;

    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "variant catalog root created");
    YVEX_TEST_ASSERT(snprintf(path, sizeof(path), "%s/candidate.gguf", root) < (int)sizeof(path),
                     "variant catalog path fits");
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    YVEX_TEST_ASSERT(fd >= 0 && ftruncate(fd, (off_t)YVEX_DEEPSEEK_DS4_FILE_BYTES) == 0 &&
                         close(fd) == 0,
                     "variant sparse extent created");
    options.path = path;
    options.readonly = 1;
    YVEX_TEST_ASSERT(yvex_artifact_open(&artifact, &options, &err) == YVEX_OK,
                     "variant sparse artifact opened");
    YVEX_TEST_ASSERT(yvex_artifact_admit_deepseek(artifact, &admission, &failure, &err) == YVEX_OK,
                     "variant catalog admission reconstructed");
    YVEX_TEST_ASSERT(admission.file_bytes == YVEX_DEEPSEEK_DS4_FILE_BYTES &&
                         admission.payload_bytes == YVEX_DEEPSEEK_DS4_PAYLOAD_BYTES &&
                         strcmp(admission.profile_identity, YVEX_DEEPSEEK_DS4_PROFILE_IDENTITY) == 0 &&
                         strcmp(admission.artifact_identity, YVEX_DEEPSEEK_DS4_ARTIFACT_IDENTITY) == 0 &&
                         strcmp(admission.admission_identity,
                                "df847745fa62e99bf2f9682af9b1fd72260365a0f69cc93a1b0c39ac034a3a51") == 0,
                     "variant catalog binds exact admitted identities");
    YVEX_TEST_ASSERT(!admission.artifact_identity_verified && admission.artifact_bytes_hashed == 0u,
                     "catalog reconstruction does not fabricate byte verification");
    yvex_artifact_close(artifact);
    YVEX_TEST_ASSERT(unlink(path) == 0 && rmdir(root) == 0,
                     "variant sparse artifact cleaned");
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
