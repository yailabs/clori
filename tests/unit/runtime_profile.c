/* Owner: tests.unit.runtime_profile.
 * Owns: profile schema, identity, timing, mutation, overflow, and stale-evidence tests.
 * Does not own: performance claims, CUDA timing, live execution, benchmark files, or rendering.
 * Invariants: deterministic fixture identities are never interpreted as measured runtime evidence.
 * Boundary: focused software tests exercise the internal profiling ABI only.
 * Purpose: prove checked profile mutation and field-wise validation before hot-path integration.
 * Inputs: bounded deterministic identities, counters, and monotonic host spans.
 * Effects: allocates no external resource and writes no file.
 * Failure: the first assertion identifies the violated profiling invariant. */
#include <yvex/internal/profile.h>
#include <limits.h>
#include <string.h>
#include "tests/test.h"
static const char identity_a[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static const char identity_b[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
static const char identity_c[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
static const char identity_d[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
static const char identity_e[] =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
static const char identity_f[] =
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
/* Purpose: construct one valid mutable profile fixture. */
static int profile_fixture(yvex_runtime_profile_record *record, yvex_error *err)
{
    return runtime_profile_begin(
        record, YVEX_RUNTIME_PROFILE_STAGES, YVEX_RUNTIME_PROFILE_GENERATION, 1u,
        identity_a, identity_b, identity_c, identity_d, identity_e, identity_f, err);
}
int yvex_test_runtime_profile(void)
{
    yvex_runtime_profile_record record, mutated;
    yvex_error err;
    YVEX_TEST_ASSERT(profile_fixture(&record, &err) == YVEX_OK,
                     "profile fixture begins");
    YVEX_TEST_ASSERT(
        runtime_profile_counter_add(
            &record, YVEX_RUNTIME_PROFILE_H2D_BYTES, 4096ull, &err) == YVEX_OK &&
        runtime_profile_phase_add(
            &record, YVEX_RUNTIME_PROFILE_ATTENTION, 9000ull, &err) == YVEX_OK,
        "profile accepts checked counters and measured phases");
    YVEX_TEST_ASSERT(runtime_profile_finish(&record, &err) == YVEX_OK &&
                         runtime_profile_validate(&record, &err) == YVEX_OK &&
                         record.counters[YVEX_RUNTIME_PROFILE_H2D_BYTES] == 4096ull &&
                         record.phase_calls[YVEX_RUNTIME_PROFILE_ATTENTION] == 1ull,
                     "profile seals and validates exact evidence");
    mutated = record;
    mutated.counters[YVEX_RUNTIME_PROFILE_H2D_BYTES]++;
    YVEX_TEST_ASSERT(runtime_profile_validate(&mutated, &err) == YVEX_ERR_FORMAT,
                     "profile mutation is rejected");
    YVEX_TEST_ASSERT(runtime_profile_counter_add(
                         &record, YVEX_RUNTIME_PROFILE_H2D_BYTES, 1ull, &err) ==
                         YVEX_ERR_STATE,
                     "sealed profile is immutable");
    YVEX_TEST_ASSERT(profile_fixture(&record, &err) == YVEX_OK,
                     "overflow fixture begins");
    record.counters[YVEX_RUNTIME_PROFILE_UPLOADS] = ULLONG_MAX;
    YVEX_TEST_ASSERT(runtime_profile_counter_add(
                         &record, YVEX_RUNTIME_PROFILE_UPLOADS, 1ull, &err) ==
                         YVEX_ERR_BOUNDS,
                     "profile counter overflow refuses");
    YVEX_TEST_ASSERT(runtime_profile_begin(
                         &record, YVEX_RUNTIME_PROFILE_DETAILED,
                         YVEX_RUNTIME_PROFILE_DECODE, 1u, "bad", identity_b,
                         identity_c, identity_d, identity_e, identity_f, &err) ==
                         YVEX_ERR_INVALID_ARG && !record.schema_version,
                     "profile malformed identity refuses without publication");
    YVEX_TEST_ASSERT(!strcmp(runtime_profile_mode_name(YVEX_RUNTIME_PROFILE_STAGES),
                             "stages") &&
                         !strcmp(runtime_profile_scope_name(YVEX_RUNTIME_PROFILE_DECODE),
                                 "decode") &&
                         !strcmp(runtime_profile_phase_name(
                                     YVEX_RUNTIME_PROFILE_ROUTED_EXPERTS),
                                 "routed_experts") &&
                         !strcmp(runtime_profile_phase_name(
                                     YVEX_RUNTIME_PROFILE_MOE_TOTAL),
                                 "moe_total") &&
                         !strcmp(runtime_profile_phase_name(
                                     YVEX_RUNTIME_PROFILE_SYNCHRONIZATION_WAIT),
                                 "synchronization_wait") &&
                         !strcmp(runtime_profile_counter_name(
                                     YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES),
                                 "kernel_launches") &&
                         !strcmp(runtime_profile_counter_name(
                                     YVEX_RUNTIME_PROFILE_DOWNLOADS),
                                 "downloads"),
                     "profile vocabulary is stable");
    return 0;
}
