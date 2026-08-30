/* Shared deterministic runner for registry-generated C test projections. */
#ifndef YVEX_TEST_RUNNER_H
#define YVEX_TEST_RUNNER_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

struct yvex_test_entry {
    const char *id;
    const char *legacy_name;
    int (*run)(void);
};

static int yvex_test_runner_token_equal(const char *token, size_t length, const char *value)
{
    return strlen(value) == length && memcmp(token, value, length) == 0;
}

static int yvex_test_runner_validate_filter(const char *filter, unsigned int *count)
{
    const char *cursor = filter;

    *count = 0u;
    while (*cursor) {
        const char *end = strchr(cursor, ',');
        const char *prior = filter;
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);

        if (length == 0u || (end && !end[1])) return 0;
        while (prior < cursor) {
            const char *prior_end = strchr(prior, ',');
            size_t prior_length = (size_t)(prior_end - prior);

            if (prior_length == length && memcmp(prior, cursor, length) == 0) return 0;
            prior = prior_end + 1;
        }
        ++*count;
        if (!end) break;
        cursor = end + 1;
    }
    return *count != 0u;
}

static int yvex_test_runner_filter_selects(const char *filter, const struct yvex_test_entry *entry)
{
    const char *cursor = filter;

    while (*cursor) {
        const char *end = strchr(cursor, ',');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);

        if (yvex_test_runner_token_equal(cursor, length, entry->id) ||
            yvex_test_runner_token_equal(cursor, length, entry->legacy_name)) {
            return 1;
        }
        if (!end) break;
        cursor = end + 1;
    }
    return 0;
}

static int yvex_test_runner_lock_workspace(void)
{
    const char *path = getenv("YVEX_TEST_WORKSPACE_LOCK");
    char default_path[128];
    struct stat directory;
    struct stat lock;
    int fd;

    if (!path || !path[0]) {
        if (stat(".", &directory) != 0 ||
            snprintf(default_path, sizeof(default_path), "/tmp/yvex-unit-%lu-%lu-%lu.lock",
                     (unsigned long)geteuid(), (unsigned long)directory.st_dev,
                     (unsigned long)directory.st_ino) >= (int)sizeof(default_path)) {
            fprintf(stderr, "FAIL: cannot derive test workspace lock path\n");
            return -1;
        }
        path = default_path;
    }
    fd = open(path, O_CREAT | O_RDWR | O_NOFOLLOW, 0600);
    if (fd < 0 || fchmod(fd, 0600) != 0 || fstat(fd, &lock) != 0 || !S_ISREG(lock.st_mode) ||
        lock.st_uid != geteuid() || flock(fd, LOCK_EX) != 0) {
        if (fd >= 0) close(fd);
        fprintf(stderr, "FAIL: cannot lock test workspace: %s\n", path);
        return -1;
    }
    return fd;
}

static int yvex_test_runner_run(const struct yvex_test_entry *entries, size_t count,
                                const char *filter_name, const char *label, int lock_workspace)
{
    const char *filter = getenv(filter_name);
    unsigned int filter_count = 0u;
    unsigned int selected_count = 0u;
    int lock_fd = -1;
    size_t i;

    if (filter && !filter[0]) filter = NULL;
    if (filter && !yvex_test_runner_validate_filter(filter, &filter_count)) {
        fprintf(stderr, "FAIL: malformed or duplicate %s\n", filter_name);
        return 1;
    }
    if (lock_workspace && !getenv("YVEX_TEST_DISABLE_WORKSPACE_LOCK")) {
        lock_fd = yvex_test_runner_lock_workspace();
        if (lock_fd < 0) return 1;
    }
    for (i = 0u; i < count; ++i) {
        int rc;

        if (filter && !yvex_test_runner_filter_selects(filter, &entries[i])) continue;
        ++selected_count;
        fprintf(stderr, "%s: %s\n", label, entries[i].id);
        rc = entries[i].run();
        if (rc != 0) {
            fprintf(stderr, "FAIL: %s exited %d\n", entries[i].id, rc);
            if (lock_fd >= 0) close(lock_fd);
            return rc;
        }
    }
    if (filter && selected_count != filter_count) {
        fprintf(stderr, "FAIL: unknown or duplicate registered %s: %s\n", filter_name, filter);
        if (lock_fd >= 0) close(lock_fd);
        return 1;
    }
    if (lock_fd >= 0) close(lock_fd);
    return 0;
}

#endif
