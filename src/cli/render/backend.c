/* Owner: src/cli/render.
 * Owns: compact backend/cuda-info formatting from typed report facts.
 * Does not own: backend admission, capability decisions, IO discovery, kernel execution, graph execution, or
 *   runtime claims.
 * Invariants: semantic status is already decided by the report owner.
 * Boundary: a rendered primitive capability is not model runtime support.
 * Purpose: provide compact backend/cuda-info formatting from typed report facts.
 * Inputs: typed domain facts, requested output mode, and caller-owned render state.
 * Effects: formats admitted facts through CLI I/O without changing domain state.
 * Failure: formatting or I/O refusal cannot alter capability facts. */
#include "src/cli/render/private.h"

#include "src/cli/io/private.h"

/* Purpose: Compute yes no for its CLI invariant (`yes_no`). */
static const char *yes_no(int value)
{
    return value ? "yes" : "no";
}

/* Contract: renders typed CUDA context and bundle admission without inference. */
/* Purpose: Render render cuda admission from typed facts (`render_cuda_admission`). */
static void render_cuda_admission(FILE *fp, const yvex_backend_report *report)
{
    yvex_cli_out_writef(fp, "context_available: %s\n",
                        yes_no(report->context_available));
    yvex_cli_out_writef(fp, "kernel_bundle: %s\n",
                        yvex_backend_bundle_admission_name(
                            report->bundle_admission));
    yvex_cli_out_writef(fp, "kernel_bundle_reason: %s\n",
                        yvex_backend_capability_reason_name(
                            report->bundle_reason));
}

/* Contract: renders the exact variant matrix already decided by the backend. */
/* Purpose: Render render variants from typed facts (`render_variants`). */
static void render_variants(FILE *fp, const yvex_backend_report *report)
{
    unsigned int i;

    yvex_cli_out_writef(fp, "variants:\n");
    for (i = 0; i < report->variant_count; ++i) {
        const yvex_backend_capability_result *result = &report->variants[i];
        yvex_cli_out_writef(fp, "  %s: %s (%s)\n",
                            yvex_backend_operation_variant_name(result->variant),
                            yvex_backend_capability_state_name(result->state),
                            yvex_backend_capability_reason_name(result->reason));
    }
}

/* Contract: renders backend capability output without probing or mutation. */
/* Purpose: Render render capabilities from typed facts (`render_capabilities`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int render_capabilities(FILE *fp, const yvex_backend_report *report)
{
    unsigned int i;

    yvex_cli_out_writef(fp, "backend: %s\n",
                        yvex_backend_kind_name(report->backend_kind));
    if (!report->available) {
        yvex_cli_out_writef(fp, "status: unsupported\nreason: %s\n",
                            report->reason);
        yvex_cli_out_writef(fp, "status: backend-unsupported\n");
        return YVEX_OK;
    }
    yvex_cli_out_writef(fp, "status: %s\n",
                        yvex_backend_status_name(report->backend_status));
    if (report->backend_kind == YVEX_BACKEND_KIND_CUDA &&
        report->has_device_info) {
        yvex_cli_out_writef(fp, "device: %d\nname: %s\ncompute_capability: %d.%d\n",
                            report->device_info.device_index,
                            report->device_info.name,
                            report->device_info.compute_capability_major,
                            report->device_info.compute_capability_minor);
        yvex_cli_out_writef(fp, "memory:\n  free_bytes: %llu\n  total_bytes: %llu\n",
                            report->device_info.free_memory_bytes,
                            report->device_info.total_memory_bytes);
    } else {
        yvex_cli_out_writef(fp, "memory:\n");
    }
    yvex_cli_out_writef(fp,
                        "  allocated_bytes: %llu\n  allocation_count: %llu\n  peak_allocated_bytes: %llu\n"
                            "capabilities:\n",
                        report->memory.allocated_bytes,
                        report->memory.allocation_count,
                        report->memory.peak_allocated_bytes);
    for (i = 0; i <= (unsigned int)YVEX_BACKEND_CAP_OP_ATTENTION; ++i) {
        yvex_cli_out_writef(fp, "  %s: %s\n",
                            yvex_backend_capability_name(
                                (yvex_backend_capability)i),
                            yes_no(report->capabilities[i]));
    }
    if (report->backend_kind == YVEX_BACKEND_KIND_CUDA) {
        render_cuda_admission(fp, report);
        render_variants(fp, report);
    }
    yvex_cli_out_writef(fp, "status: backend-capabilities\n");
    return YVEX_OK;
}

/* Contract: renders CUDA device facts and typed bundle admission only. */
/* Purpose: Render render cuda info from typed facts (`render_cuda_info`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
static int render_cuda_info(FILE *fp, const yvex_backend_report *report)
{
    if (!report->available) {
        yvex_cli_out_writef(fp, "cuda: unavailable\nreason: %s\n",
                            report->reason);
        yvex_cli_out_writef(fp, "status: cuda-unavailable\n");
        return YVEX_OK;
    }
    yvex_cli_out_writef(fp, "cuda: available\ndevice_count: >=1\n\n");
    yvex_cli_out_writef(fp,
                        "device %d:\n  name: %s\n  compute_capability: %d.%d\n  global_memory_bytes: %llu\n"
                            "  free_memory_bytes: %llu\n  total_memory_bytes: %llu\n  unified_addressing: %s\n "
                            " managed_memory: %s\n",
                        report->device_info.device_index,
                        report->device_info.name,
                        report->device_info.compute_capability_major,
                        report->device_info.compute_capability_minor,
                        report->device_info.global_memory_bytes,
                        report->device_info.free_memory_bytes,
                        report->device_info.total_memory_bytes,
                        yes_no(report->device_info.unified_addressing),
                        yes_no(report->device_info.managed_memory));
    render_cuda_admission(fp, report);
    yvex_cli_out_writef(fp, "\nstatus: cuda-info\n");
    return YVEX_OK;
}

/* Purpose: Render backend render from typed facts (`yvex_backend_render`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_backend_render(FILE *fp, const yvex_backend_report *report)
{
    if (!fp || !report) return YVEX_ERR_INVALID_ARG;
    return report->kind == YVEX_BACKEND_REPORT_CUDA_INFO
               ? render_cuda_info(fp, report)
               : render_capabilities(fp, report);
}

/* Purpose: Render backend render help from typed facts (`yvex_backend_render_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_backend_render_help(FILE *fp)
{
    yvex_cli_out_writef(
        fp, "usage: yvex evidence backend cpu|cuda\n\n"
            "Reports context, bundle, and exact primitive capabilities.\n");
    return YVEX_OK;
}

/* Purpose: Render cuda info render help from typed facts (`yvex_cuda_info_render_help`).
 * Inputs: Borrowed typed facts.
 * Effects: Writes through CLI I/O only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: No capability policy. */
int yvex_cuda_info_render_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
                        "usage: yvex evidence cuda\n\nReports CUDA driver, device, context, and bundle facts.\n");
    return YVEX_OK;
}
