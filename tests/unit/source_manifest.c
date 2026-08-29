/*
 * Verifies open-weight intake source provenance scanning and manifest writing over a tiny fake
 * source tree. No external model files are required or committed.
 */
#include "tests/test.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/source.h>

static int make_dir(const char *path)
{
    if (mkdir(path, 0777) != 0 && errno != EEXIST) {
        return 0;
    }
    return 1;
}

static int write_text(const char *path, const char *text)
{
    FILE *fp = fopen(path, "wb");

    if (!fp) {
        return 0;
    }
    fputs(text, fp);
    return fclose(fp) == 0;
}

static int read_file(const char *path, char *buf, size_t cap)
{
    FILE *fp = fopen(path, "rb");
    size_t n;

    if (!fp || cap == 0) {
        if (fp) {
            fclose(fp);
        }
        return 0;
    }
    n = fread(buf, 1, cap - 1, fp);
    buf[n] = '\0';
    fclose(fp);
    return 1;
}

static int contains_text(const char *haystack, const char *needle)
{
    return strstr(haystack, needle) != NULL;
}

int yvex_test_source_manifest(void)
{
    const char *root = "build/tests/source_manifest_fixture";
    const char *manifest = "build/tests/source_manifest_fixture/manifest.json";
    yvex_source_manifest_summary summary;
    yvex_source_manifest_options options;
    yvex_error err;
    char json[8192];
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/source_manifest_fixture") == 0,
                     "clear source manifest fixture");
    YVEX_TEST_ASSERT(make_dir("build"), "make build");
    YVEX_TEST_ASSERT(make_dir("build/tests"), "make build/tests");
    YVEX_TEST_ASSERT(make_dir(root), "make source manifest fixture");
    YVEX_TEST_ASSERT(write_text("build/tests/source_manifest_fixture/config.json", "{}\n"), "write config");
    YVEX_TEST_ASSERT(write_text("build/tests/source_manifest_fixture/tokenizer.json", "{}\n"), "write tokenizer");
    YVEX_TEST_ASSERT(write_text("build/tests/source_manifest_fixture/README.md", "fixture\n"), "write readme");
    YVEX_TEST_ASSERT(write_text("build/tests/source_manifest_fixture/model-00001-of-00002.safetensors", "abc"), "write safetensors 1");
    YVEX_TEST_ASSERT(write_text("build/tests/source_manifest_fixture/model-00002-of-00002.safetensors", "defg"), "write safetensors 2");

    yvex_error_clear(&err);
    memset(&summary, 0, sizeof(summary));
    rc = yvex_source_manifest_scan_local(root, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "scan local path succeeds");
    YVEX_TEST_ASSERT(summary.file_count == 5, "scan counts files");
    YVEX_TEST_ASSERT(summary.safetensors_count == 2, "scan counts safetensors");
    YVEX_TEST_ASSERT(summary.total_size_bytes > 0, "scan totals size");
    YVEX_TEST_ASSERT(summary.has_config, "scan detects config");
    YVEX_TEST_ASSERT(summary.has_tokenizer, "scan detects tokenizer");
    YVEX_TEST_ASSERT(summary.has_safetensors, "scan detects safetensors");

    memset(&options, 0, sizeof(options));
    options.repo = "test-org/test-model";
    options.revision = "test-rev";
    options.license = "test-license";
    options.model_card = "https://example.invalid/test-model";
    options.local_path = root;
    options.node_name = "test-node";
    options.status = YVEX_SOURCE_STATUS_IN_PROGRESS;
    options.include_files = 1;

    memset(&summary, 0, sizeof(summary));
    rc = yvex_source_manifest_write_json(manifest, &options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write manifest JSON");
    YVEX_TEST_ASSERT(read_file(manifest, json, sizeof(json)), "manifest file exists");
    YVEX_TEST_ASSERT(contains_text(json, "\"schema\": \"yvex.source_manifest.v1\""), "manifest has schema");
    YVEX_TEST_ASSERT(contains_text(json, "\"repo\": \"test-org/test-model\""), "manifest has repo");
    YVEX_TEST_ASSERT(contains_text(json, "\"path\": \"build/tests/source_manifest_fixture\""), "manifest has local path");
    YVEX_TEST_ASSERT(contains_text(json, "\"status\": \"in-progress\""), "manifest has status");
    YVEX_TEST_ASSERT(contains_text(json, "\"path\": \"config.json\""), "manifest lists relative config path");
    YVEX_TEST_ASSERT(contains_text(json, "\"path\": \"model-00001-of-00002.safetensors\""), "manifest lists relative safetensors path");
    YVEX_TEST_ASSERT(contains_text(json, "\"sha256\": null"), "manifest uses null sha256");

    rc = yvex_source_manifest_scan_local("build/tests/source_manifest_fixture/missing", &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "missing local path returns error");

    rc = yvex_source_manifest_write_json(NULL, &options, NULL, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG, "invalid args return error");

    YVEX_TEST_ASSERT_STREQ(yvex_source_status_name(YVEX_SOURCE_STATUS_COMPLETE), "complete", "source status name");
    return 0;
}
