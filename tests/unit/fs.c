/*
 * Exercises runtime filesystem runtime filesystem path resolution, project-local paths, run ID
 * creation, run-directory preparation, and directory creation behavior.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <yvex/core.h>
#include <yvex/internal/core.h>

#include "tests/test.h"

static int ensure_dir(const char *path)
{
    if (mkdir(path, 0777) == 0 || errno == EEXIST) {
        return 0;
    }
    fprintf(stderr, "FAIL: mkdir %s: %s\n", path, strerror(errno));
    return 1;
}

static int ensure_test_root(void)
{
    return ensure_dir("build") || ensure_dir("build/tests") || ensure_dir("build/tests/tmp");
}

static char *make_temp_dir(char *template_path)
{
    if (ensure_test_root() != 0) {
        return NULL;
    }
    return mkdtemp(template_path);
}

static void clear_yvex_env(void)
{
    unsetenv("YVEX_CONFIG_DIR");
    unsetenv("YVEX_CACHE_DIR");
    unsetenv("YVEX_STATE_DIR");
    unsetenv("YVEX_DATA_DIR");
    unsetenv("YVEX_RUN_DIR");
}

static int path_is_dir(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) {
        return 0;
    }
    return S_ISDIR(st.st_mode) ? 1 : 0;
}

static int test_default_paths(void)
{
    char home_template[] = "build/tests/tmp/homeXXXXXX";
    char expected[YVEX_PATH_CAP * 2];
    char *home;
    yvex_paths paths;
    yvex_error err;
    int rc;

    clear_yvex_env();
    home = make_temp_dir(home_template);
    YVEX_TEST_ASSERT(home != NULL, "mkdtemp home");
    setenv("HOME", home, 1);

    rc = yvex_paths_default(&paths, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "default paths ok");

    snprintf(expected, sizeof(expected), "%s/.config/yvex", home);
    YVEX_TEST_ASSERT_STREQ(paths.config_dir, expected, "default config");
    snprintf(expected, sizeof(expected), "%s/.cache/yvex", home);
    YVEX_TEST_ASSERT_STREQ(paths.cache_dir, expected, "default cache");
    snprintf(expected, sizeof(expected), "%s/.local/state/yvex", home);
    YVEX_TEST_ASSERT_STREQ(paths.state_dir, expected, "default state");
    snprintf(expected, sizeof(expected), "%s/.local/share/yvex", home);
    YVEX_TEST_ASSERT_STREQ(paths.data_dir, expected, "default data");
    YVEX_TEST_ASSERT_STREQ(paths.project_dir, "", "default project empty");
    return 0;
}

static int test_env_overrides(void)
{
    char home_template[] = "build/tests/tmp/envhomeXXXXXX";
    char *home;
    yvex_paths paths;
    yvex_error err;
    int rc;

    clear_yvex_env();
    home = make_temp_dir(home_template);
    YVEX_TEST_ASSERT(home != NULL, "mkdtemp env home");
    setenv("HOME", home, 1);
    setenv("YVEX_CONFIG_DIR", "/tmp/yvex-config-test", 1);
    setenv("YVEX_CACHE_DIR", "/tmp/yvex-cache-test", 1);
    setenv("YVEX_STATE_DIR", "/tmp/yvex-state-test", 1);
    setenv("YVEX_DATA_DIR", "/tmp/yvex-data-test", 1);

    rc = yvex_paths_default(&paths, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "env override paths ok");
    YVEX_TEST_ASSERT_STREQ(paths.config_dir, "/tmp/yvex-config-test", "env config");
    YVEX_TEST_ASSERT_STREQ(paths.cache_dir, "/tmp/yvex-cache-test", "env cache");
    YVEX_TEST_ASSERT_STREQ(paths.state_dir, "/tmp/yvex-state-test", "env state");
    YVEX_TEST_ASSERT_STREQ(paths.data_dir, "/tmp/yvex-data-test", "env data");
    return 0;
}

static int test_missing_home_and_bounds(void)
{
    char long_home[YVEX_PATH_CAP + 256];
    yvex_paths paths;
    yvex_error err;
    int rc;
    size_t i;

    clear_yvex_env();
    unsetenv("HOME");
    rc = yvex_paths_default(&paths, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE, "missing HOME returns state");
    YVEX_TEST_ASSERT_STREQ(yvex_error_where(&err), "yvex_paths_default", "missing HOME where");

    for (i = 0; i < sizeof(long_home) - 1; ++i) {
        long_home[i] = 'x';
    }
    long_home[sizeof(long_home) - 1] = '\0';
    setenv("HOME", long_home, 1);
    rc = yvex_paths_default(&paths, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "long HOME returns bounds");
    return 0;
}

static int test_project_paths(void)
{
    yvex_paths paths;
    yvex_error err;
    int rc;

    clear_yvex_env();
    rc = yvex_paths_project(&paths, ".", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "project paths ok");
    YVEX_TEST_ASSERT_STREQ(paths.project_dir, "./.yvex", "project dir");
    YVEX_TEST_ASSERT_STREQ(paths.config_dir, "./.yvex", "project config dir");
    YVEX_TEST_ASSERT_STREQ(paths.cache_dir, "./.yvex/cache", "project cache dir");
    YVEX_TEST_ASSERT_STREQ(paths.state_dir, "./.yvex/state", "project state dir");
    YVEX_TEST_ASSERT_STREQ(paths.data_dir, "./.yvex/data", "project data dir");
    return 0;
}

static int test_run_id_and_prepare(void)
{
    char project_template[] = "build/tests/tmp/projectXXXXXX";
    char expected[YVEX_PATH_CAP * 2];
    char run_id[YVEX_RUN_ID_CAP];
    char *project;
    yvex_paths paths;
    yvex_run_dir run;
    yvex_error err;
    int rc;

    clear_yvex_env();
    project = make_temp_dir(project_template);
    YVEX_TEST_ASSERT(project != NULL, "mkdtemp project");

    rc = yvex_run_id_make(run_id, sizeof(run_id), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "run id ok");
    YVEX_TEST_ASSERT(strncmp(run_id, "run_", 4) == 0, "run id prefix");

    rc = yvex_paths_project(&paths, project, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "project paths for run");
    rc = yvex_run_dir_prepare(&run, &paths, "run_fixed", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "run dir prepare");

    snprintf(expected, sizeof(expected), "%s/.yvex/runs/run_fixed", project);
    YVEX_TEST_ASSERT_STREQ(run.root, expected, "run root");
    snprintf(expected, sizeof(expected), "%s/command.txt", run.root);
    YVEX_TEST_ASSERT_STREQ(run.command_path, expected, "command path");
    snprintf(expected, sizeof(expected), "%s/stdout.log", run.root);
    YVEX_TEST_ASSERT_STREQ(run.stdout_path, expected, "stdout path");
    snprintf(expected, sizeof(expected), "%s/stderr.log", run.root);
    YVEX_TEST_ASSERT_STREQ(run.stderr_path, expected, "stderr path");
    snprintf(expected, sizeof(expected), "%s/metrics.json", run.root);
    YVEX_TEST_ASSERT_STREQ(run.metrics_path, expected, "metrics path");
    snprintf(expected, sizeof(expected), "%s/trace.jsonl", run.root);
    YVEX_TEST_ASSERT_STREQ(run.trace_path, expected, "trace path");
    snprintf(expected, sizeof(expected), "%s/receipt.json", run.root);
    YVEX_TEST_ASSERT_STREQ(run.receipt_path, expected, "receipt path");
    return 0;
}

static int test_run_dir_create(void)
{
    char project_template[] = "build/tests/tmp/createXXXXXX";
    char *project;
    yvex_paths paths;
    yvex_run_dir run;
    yvex_run_dir collision;
    yvex_error err;
    FILE *fp;
    int rc;

    clear_yvex_env();
    project = make_temp_dir(project_template);
    YVEX_TEST_ASSERT(project != NULL, "mkdtemp create project");

    rc = yvex_paths_project(&paths, project, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "project paths create");
    rc = yvex_run_dir_prepare(&run, &paths, "run_create", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "run prepare create");
    rc = yvex_run_dir_create(&run, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "run create");
    YVEX_TEST_ASSERT(path_is_dir(run.root), "run root is directory");
    rc = yvex_run_dir_create(&run, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "run create tolerates existing directory");

    rc = yvex_run_dir_prepare(&collision, &paths, "run_collision", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "collision prepare");
    rc = yvex_run_dir_create(&run, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "ensure parent exists");
    fp = fopen(collision.root, "w");
    YVEX_TEST_ASSERT(fp != NULL, "create collision file");
    fclose(fp);
    rc = yvex_run_dir_create(&collision, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO, "file collision rejected");
    YVEX_TEST_ASSERT(yvex_error_is_set(&err), "file collision sets error");
    return 0;
}

static int reject_file_candidate(int descriptor, size_t count,
                                 void *context, yvex_error *err)
{
    unsigned char byte = 0u;
    int *called = (int *)context;

    if (called) *called += 1;
    if (descriptor < 0 || !count || pread(descriptor, &byte, 1u, 0) != 1) {
        yvex_error_set(err, YVEX_ERR_IO, "test.file_validator",
                       "candidate could not be read through its reopened descriptor");
        return YVEX_ERR_IO;
    }
    yvex_error_set(err, YVEX_ERR_FORMAT, "test.file_validator",
                   "candidate rejection was requested");
    return YVEX_ERR_FORMAT;
}

static int test_safe_file_lifecycle(void)
{
    char root_template[] = "build/tests/tmp/safeXXXXXX";
    char path[YVEX_PATH_CAP], link_path[YVEX_PATH_CAP], unsafe[YVEX_PATH_CAP];
    char close_path[YVEX_PATH_CAP], unlink_path[YVEX_PATH_CAP];
    char cleanup_path[YVEX_PATH_CAP], validate_path[YVEX_PATH_CAP];
    char temporary[YVEX_PATH_CAP];
    const unsigned char expected[] = "canonical-bytes";
    unsigned char *actual = NULL;
    size_t actual_count = 0u;
    yvex_core_file_faults faults;
    yvex_core_file_result result;
    yvex_error err;
    char *root;
    int validator_called = 0, rc;

    root = make_temp_dir(root_template);
    YVEX_TEST_ASSERT(root != NULL, "mkdtemp safe file root");
    YVEX_TEST_ASSERT(snprintf(path, sizeof(path), "%s/record", root) < (int)sizeof(path),
                     "safe file path fits");
    rc = yvex_core_file_publish_noreplace(
        path, expected, sizeof(expected) - 1u, NULL, NULL, NULL, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && result.actual == sizeof(expected) - 1u,
                     "safe file publishes exact bytes");
    rc = yvex_core_file_read_snapshot(path, 64u, &actual, &actual_count, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && actual_count == sizeof(expected) - 1u &&
                         memcmp(actual, expected, actual_count) == 0,
                     "safe file reopens one stable snapshot");
    free(actual);
    actual = NULL;
    rc = yvex_core_file_publish_noreplace(
        path, expected, sizeof(expected) - 1u, NULL, NULL, NULL, &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE &&
                         result.stage == YVEX_CORE_FILE_STAGE_CONFLICT,
                     "safe file refuses destination replacement");
    YVEX_TEST_ASSERT(snprintf(link_path, sizeof(link_path), "%s-link", root) <
                         (int)sizeof(link_path) &&
                         symlink(root, link_path) == 0 &&
                         snprintf(unsafe, sizeof(unsafe), "%s/record", link_path) <
                             (int)sizeof(unsafe),
                     "safe file symlink fixture is bounded");
    rc = yvex_core_file_read_snapshot(unsafe, 64u, &actual, &actual_count, &result, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK && result.stage == YVEX_CORE_FILE_STAGE_PATH,
                     "safe file refuses symlink parent traversal");
    YVEX_TEST_ASSERT(snprintf(close_path, sizeof(close_path), "%s/close-failure", root) <
                             (int)sizeof(close_path) &&
                         snprintf(unlink_path, sizeof(unlink_path), "%s/unlink-failure", root) <
                             (int)sizeof(unlink_path) &&
                         snprintf(cleanup_path, sizeof(cleanup_path), "%s/cleanup-failure", root) <
                             (int)sizeof(cleanup_path) &&
                         snprintf(validate_path, sizeof(validate_path), "%s/validate-failure", root) <
                             (int)sizeof(validate_path),
                     "fault-injection paths fit");

    rc = yvex_core_file_publish_noreplace(
        validate_path, expected, sizeof(expected) - 1u, NULL,
        reject_file_candidate, &validator_called, &result, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_FORMAT && validator_called == 1 &&
            result.stage == YVEX_CORE_FILE_STAGE_VALIDATE &&
            result.cleanup_stage == YVEX_CORE_FILE_CLEANUP_NONE &&
            access(validate_path, F_OK) != 0,
        "candidate validation refusal leaves no published destination");

    memset(&faults, 0, sizeof(faults));
    faults.inject_file_close_failure = 1;
    rc = yvex_core_file_publish_noreplace(
        close_path, expected, sizeof(expected) - 1u, &faults, NULL, NULL,
        &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO && result.stage == YVEX_CORE_FILE_STAGE_FILE_CLOSE &&
                         result.cleanup_stage == YVEX_CORE_FILE_CLEANUP_NONE &&
                         access(close_path, F_OK) != 0,
                     "injected close failure withdraws the unpublished destination");

    memset(&faults, 0, sizeof(faults));
    faults.inject_temporary_unlink_failure = 1;
    rc = yvex_core_file_publish_noreplace(
        unlink_path, expected, sizeof(expected) - 1u, &faults, NULL, NULL,
        &result, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO &&
                         result.stage == YVEX_CORE_FILE_STAGE_TEMPORARY_UNLINK &&
                         result.cleanup_stage == YVEX_CORE_FILE_CLEANUP_NONE &&
                         access(unlink_path, F_OK) != 0,
                     "temporary unlink failure rolls back the linked destination");

    memset(&faults, 0, sizeof(faults));
    faults.inject_file_close_failure = 1;
    faults.inject_cleanup_unlink_failure = 1;
    rc = yvex_core_file_publish_noreplace(
        cleanup_path, expected, sizeof(expected) - 1u, &faults, NULL, NULL,
        &result, &err);
    YVEX_TEST_ASSERT(snprintf(temporary, sizeof(temporary), "%s/.cleanup-failure.%llu.0.tmp",
                              root, (unsigned long long)getpid()) < (int)sizeof(temporary) &&
                         rc == YVEX_ERR_IO &&
                         result.stage == YVEX_CORE_FILE_STAGE_FILE_CLOSE &&
                         result.cleanup_stage == YVEX_CORE_FILE_CLEANUP_TEMPORARY_UNLINK &&
                         result.cleanup_system_error == EIO &&
                         access(cleanup_path, F_OK) != 0 && access(temporary, F_OK) != 0,
                     "cleanup failure retains both errors after removing the exact temporary");
    YVEX_TEST_ASSERT(unlink(link_path) == 0 && unlink(path) == 0 && rmdir(root) == 0,
                     "safe file fixture removes only owned paths");
    return 0;
}

int yvex_test_fs(void)
{
    if (test_default_paths() != 0) {
        return 1;
    }
    if (test_env_overrides() != 0) {
        return 1;
    }
    if (test_missing_home_and_bounds() != 0) {
        return 1;
    }
    if (test_project_paths() != 0) {
        return 1;
    }
    if (test_run_id_and_prepare() != 0) {
        return 1;
    }
    if (test_run_dir_create() != 0) {
        return 1;
    }
    if (test_safe_file_lifecycle() != 0) {
        return 1;
    }
    return 0;
}
