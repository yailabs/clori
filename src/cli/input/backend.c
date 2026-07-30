/* Owner: src/cli/input.
 * Owns: backend/cuda-info grammar and typed request construction.
 * Does not own: backend discovery, capability policy, rendering, or execution.
 * Invariants: argv is borrowed and parsing has no backend side effects.
 * Boundary: valid grammar is not CUDA operation support.
 * Purpose: provide backend/cuda-info grammar and typed request construction.
 * Inputs: bounded command arguments and caller-owned typed request storage.
 * Effects: publishes request fields only after complete grammar validation.
 * Failure: invalid or ambiguous grammar leaves the request uncommitted. */
#include "src/cli/input/private.h"

#include <string.h>

/* Contract: parses `yvex evidence backend cpu|cuda` without opening a backend. */

/* Purpose: Admit the two backend names supported by CLI selectors.
 * Inputs: Borrowed nullable backend name.
 * Effects: None.
 * Failure: Returns false for null or unrecognized names.
 * Boundary: Name admission does not open or prove a backend. */
int cli_backend_name_valid(const char *name)
{
    return name && (strcmp(name, "cpu") == 0 || strcmp(name, "cuda") == 0);
}

/* Purpose: Parse backend args parse into typed CLI state (`yvex_backend_args_parse`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_backend_args_parse(int argc, char **argv,
                            yvex_backend_args *out, yvex_error *err)
{
    if (!out || !argv) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "backend",
                       "backend arguments are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->request.kind = YVEX_BACKEND_REPORT_CAPABILITIES;
    if (argc == 3 && (strcmp(argv[2], "--help") == 0 ||
                      strcmp(argv[2], "-h") == 0)) {
        out->help = 1;
        return YVEX_OK;
    }
    if (argc != 3) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "backend",
                       "backend requires cpu or cuda");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(argv[2], "cpu") == 0) {
        out->request.backend_kind = YVEX_BACKEND_KIND_CPU;
    } else if (strcmp(argv[2], "cuda") == 0) {
        out->request.backend_kind = YVEX_BACKEND_KIND_CUDA;
    } else {
        yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "backend",
                        "unknown backend kind: %s", argv[2]);
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Contract: parses the option-free `yvex cuda-info` diagnostic request. */
/* Purpose: Parse cuda info args parse into typed CLI state (`yvex_cuda_info_args_parse`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_cuda_info_args_parse(int argc, char **argv,
                              yvex_backend_args *out, yvex_error *err)
{
    if (!out || !argv) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda-info",
                       "cuda-info arguments are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->request.kind = YVEX_BACKEND_REPORT_CUDA_INFO;
    out->request.backend_kind = YVEX_BACKEND_KIND_CUDA;
    if (argc == 3 && (strcmp(argv[2], "--help") == 0 ||
                      strcmp(argv[2], "-h") == 0)) {
        out->help = 1;
        return YVEX_OK;
    }
    if (argc != 2) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "cuda-info",
                       "usage: yvex cuda-info");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
