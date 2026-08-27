/* Deployment capacity reflects current host and cgroup pressure, never package identity. */
#include "src/runtime/private.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/deployment.h>

static int runtime_capacity_value(const char *text, unsigned long long *value)
{
    char *tail = NULL;
    unsigned long long parsed;
    if (!text || !value || !text[0] || text[0] == '-' || !strncmp(text, "max", 3u))
        return 0;
    errno = 0;
    parsed = strtoull(text, &tail, 10);
    if (errno || tail == text) return 0;
    while (*tail == ' ' || *tail == '\t' || *tail == '\r' || *tail == '\n') ++tail;
    if (*tail) return 0;
    *value = parsed;
    return 1;
}

/* Select the tightest remaining cgroup-v2 memory extent across the process hierarchy. */
static int runtime_cgroup_memory(unsigned long long *capacity,
                                 unsigned long long *available)
{
    const char *injected = getenv("YVEX_TEST_RUNTIME_CGROUP_AVAILABLE_MEMORY_BYTES");
    static const char *const controls[] = {"memory.max", "memory.high"};
    const char *root = "/sys/fs/cgroup";
    char group[PATH_MAX], directory[PATH_MAX], path[PATH_MAX], text[128];
    char *relative, *newline, *slash;
    unsigned long long tightest_capacity = ULLONG_MAX;
    unsigned long long tightest_available = ULLONG_MAX;
    size_t control, root_length = strlen(root);
    int found = 0, written;
    if (!capacity || !available) return -1;
    if (injected) {
        if (!runtime_capacity_value(injected, available)) return -1;
        *capacity = ULLONG_MAX;
        return 1;
    }
    if (!yvex_core_file_read_text_prefix("/proc/self/cgroup", group, sizeof(group)) ||
        !(relative = strstr(group, "0::")) ||
        (relative != group && relative[-1] != '\n') ||
        relative[3] != '/' || strstr(relative + 3, ".."))
        return 0;
    relative += 3;
    if ((newline = strchr(relative, '\n'))) *newline = '\0';
    written = snprintf(directory, sizeof(directory), "%s%s", root, relative);
    if (written <= 0 || (size_t)written >= sizeof(directory)) return -1;
    for (;;) {
        unsigned long long current;
        int current_known;
        written = snprintf(path, sizeof(path), "%s/memory.current", directory);
        if (written <= 0 || (size_t)written >= sizeof(path)) return -1;
        current_known = yvex_core_file_read_text_prefix(path, text, sizeof(text)) &&
                        runtime_capacity_value(text, &current);
        for (control = 0u; current_known && control < 2u; ++control) {
            unsigned long long limit, remaining;
            written = snprintf(path, sizeof(path), "%s/%s", directory, controls[control]);
            if (written <= 0 || (size_t)written >= sizeof(path)) return -1;
            if (!yvex_core_file_read_text_prefix(path, text, sizeof(text)) ||
                !runtime_capacity_value(text, &limit))
                continue;
            remaining = current < limit ? limit - current : 0ull;
            if (!found || limit < tightest_capacity) tightest_capacity = limit;
            if (!found || remaining < tightest_available)
                tightest_available = remaining;
            found = 1;
        }
        if (!strcmp(directory, root)) break;
        slash = strrchr(directory, '/');
        if (!slash || (size_t)(slash - directory) < root_length) return -1;
        *slash = '\0';
    }
    if (found) {
        *capacity = tightest_capacity;
        *available = tightest_available;
    }
    return found;
}

static int runtime_system_memory(unsigned long long *total,
                                 unsigned long long *available)
{
    const char *injected_total = getenv("YVEX_TEST_RUNTIME_TOTAL_MEMORY_BYTES");
    const char *injected_available =
        getenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES");
    char line[128];
    FILE *meminfo;
    unsigned long long system_total = 0ull, system_available = 0ull;
    long total_pages, available_pages, page_bytes;

    if (!total || !available) return 0;
    if (injected_total && !runtime_capacity_value(injected_total, &system_total)) return 0;
    if (injected_available &&
        !runtime_capacity_value(injected_available, &system_available))
        return 0;
    if ((!system_total || !system_available) &&
        (meminfo = fopen("/proc/meminfo", "r"))) {
        while (fgets(line, sizeof(line), meminfo)) {
            unsigned long long value;
            if (!system_total && sscanf(line, "MemTotal: %llu kB", &value) == 1)
                if (!yvex_core_u64_mul(value, 1024ull, &system_total))
                    system_total = 0ull;
            if (!system_available &&
                sscanf(line, "MemAvailable: %llu kB", &value) == 1)
                if (!yvex_core_u64_mul(value, 1024ull, &system_available))
                    system_available = 0ull;
        }
        if (fclose(meminfo) != 0) return 0;
    }
    if (!system_total || !system_available) {
#if defined(_SC_PHYS_PAGES) && defined(_SC_AVPHYS_PAGES)
        total_pages = sysconf(_SC_PHYS_PAGES);
        available_pages = sysconf(_SC_AVPHYS_PAGES);
        page_bytes = sysconf(_SC_PAGESIZE);
        if (total_pages <= 0 || available_pages <= 0 || page_bytes <= 0 ||
            (!system_total &&
             !yvex_core_u64_mul((unsigned long long)total_pages,
                                (unsigned long long)page_bytes, &system_total)) ||
            (!system_available &&
             !yvex_core_u64_mul((unsigned long long)available_pages,
                                (unsigned long long)page_bytes,
                                &system_available)))
            return 0;
#else
        (void)total_pages;
        (void)available_pages;
        (void)page_bytes;
        return 0;
#endif
    }
    if (system_available > system_total) system_available = system_total;
    *total = system_total;
    *available = system_available;
    return 1;
}

int yvex_runtime_private_memory_capacity(unsigned long long *total_bytes,
                                         unsigned long long *available_bytes,
                                         int *process_limited)
{
    unsigned long long total, available, process_capacity, process_available;
    int cgroup;

    if (!total_bytes || !available_bytes || !process_limited ||
        !runtime_system_memory(&total, &available))
        return 0;
    *process_limited = 0;
    cgroup = runtime_cgroup_memory(&process_capacity, &process_available);
    if (cgroup < 0) return 0;
    if (cgroup > 0) {
        int limited = process_capacity < total || process_available < available;
        if (process_capacity < total) total = process_capacity;
        if (process_available < available) available = process_available;
        *process_limited = limited;
    }
    if (available > total) available = total;
    *total_bytes = total;
    *available_bytes = available;
    return 1;
}

unsigned long long yvex_runtime_private_system_reserve(
    unsigned long long capacity_bytes)
{
    unsigned long long proportional = capacity_bytes / 8ull;
    return proportional > YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE
               ? proportional
               : YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE;
}
