/*
 * Provide typed backend/cuda-info parse, report, render, and exit-code dispatch.
 *
 * Command dispatch never derives capability from context status. Capability inspection is bounded
 * backend evidence, not runtime.
 */
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include "src/cli/render/private.h"

static int backend_cli_run(const yvex_backend_args *args) {
    yvex_backend_report report;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_backend_report_build(&args->request, &report, &err);
    if (rc != YVEX_OK) {
        return print_yvex_error(&err, rc == YVEX_ERR_NOMEM ? 3 : exit_for_status(rc));
    }
    (void)yvex_backend_render(yvex_cli_out_stdout(), &report);
    return report.exit_code;
}

int yvex_backend_command(int argc, char **argv) {
    yvex_backend_args args;
    yvex_error err;
    int rc = yvex_backend_args_parse(argc, argv, &args, &err);

    if (rc != YVEX_OK)
        return print_yvex_error(&err, rc == YVEX_ERR_NOMEM ? 3 : exit_for_status(rc));
    if (args.help) {
        (void)yvex_backend_render_help(yvex_cli_out_stdout());
        return 0;
    }
    return backend_cli_run(&args);
}

int yvex_cuda_info_command(int argc, char **argv) {
    yvex_backend_args args;
    yvex_error err;
    int rc = yvex_cuda_info_args_parse(argc, argv, &args, &err);

    if (rc != YVEX_OK)
        return print_yvex_error(&err, rc == YVEX_ERR_NOMEM ? 3 : exit_for_status(rc));
    if (args.help) {
        (void)yvex_cuda_info_render_help(yvex_cli_out_stdout());
        return 0;
    }
    return backend_cli_run(&args);
}

void yvex_backend_help(FILE *fp) {
    (void)yvex_backend_render_help(fp);
}

void yvex_cuda_info_help(FILE *fp) {
    (void)yvex_cuda_info_render_help(fp);
}
