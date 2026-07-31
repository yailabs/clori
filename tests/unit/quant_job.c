
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

static int write_file(const char *path, const char *text)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fputs(text, fp);
    return fclose(fp) == 0;
}

static int test_names(void)
{
    YVEX_TEST_ASSERT_STREQ(yvex_quant_job_status_name(YVEX_QUANT_JOB_STATUS_READY), "ready", "status name");
    YVEX_TEST_ASSERT_STREQ(yvex_quant_job_tool_name(YVEX_QUANT_JOB_TOOL_EXTERNAL), "external", "tool name");
    YVEX_TEST_ASSERT(yvex_quant_job_status_from_name("succeeded") == YVEX_QUANT_JOB_STATUS_SUCCEEDED, "status parse");
    YVEX_TEST_ASSERT(yvex_quant_job_tool_from_name("external") == YVEX_QUANT_JOB_TOOL_EXTERNAL, "tool parse");
    YVEX_TEST_ASSERT(yvex_quant_job_tool_from_name("yvex-internal") == YVEX_QUANT_JOB_TOOL_YVEX_INTERNAL, "internal tool parse");
    YVEX_TEST_ASSERT(yvex_quant_job_tool_from_name("unknown") == YVEX_QUANT_JOB_TOOL_UNKNOWN, "unknown tool parse");
    YVEX_TEST_ASSERT(yvex_quant_job_tool_from_name("missing") == YVEX_QUANT_JOB_TOOL_UNKNOWN, "unknown tool");
    return 0;
}

static void fill_options(yvex_quant_job_options *options)
{
    memset(options, 0, sizeof(*options));
    options->name = "test-job";
    options->architecture = "deepseek4";
    options->tool_path = "build/tests/quant-job/tool";
    options->source_manifest_path = "build/tests/quant-job/source-manifest.json";
    options->native_source_dir = "build/tests/quant-job/native";
    options->template_path = "build/tests/quant-job/template.gguf";
    options->quant_policy_path = "build/tests/quant-job/policy.json";
    options->imatrix_manifest_path = "build/tests/quant-job/imatrix.json";
    options->imatrix_path = "build/tests/quant-job/imatrix.dat";
    options->out_gguf_path = "build/tests/quant-job/out.gguf";
    options->log_path = "build/tests/quant-job/job.log";
    options->command = "test command";
    options->tool = YVEX_QUANT_JOB_TOOL_EXTERNAL;
    options->status = YVEX_QUANT_JOB_STATUS_READY;
}

static int test_write_validate_ready(void)
{
    yvex_quant_job_options options;
    yvex_quant_job_summary summary;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/quant-job && mkdir -p build/tests/quant-job/native") == 0,
                     "mkdir quant job");
    YVEX_TEST_ASSERT(write_file("build/tests/quant-job/tool", "#!/bin/sh\nexit 0\n"), "write tool");
    YVEX_TEST_ASSERT(system("chmod +x build/tests/quant-job/tool") == 0, "chmod tool");
    YVEX_TEST_ASSERT(write_file("build/tests/quant-job/source-manifest.json", "{}\n"), "write source");
    YVEX_TEST_ASSERT(write_file("build/tests/quant-job/template.gguf", "fake\n"), "write template");
    YVEX_TEST_ASSERT(write_file("build/tests/quant-job/policy.json", "{}\n"), "write policy");
    YVEX_TEST_ASSERT(write_file("build/tests/quant-job/imatrix.dat", "fake\n"), "write imatrix");

    fill_options(&options);
    yvex_error_clear(&err);
    rc = yvex_quant_job_write_json("build/tests/quant-job/job.json", &options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write manifest");
    YVEX_TEST_ASSERT(summary.tool_exists == 1, "tool exists");
    YVEX_TEST_ASSERT(summary.source_exists == 1, "source exists");
    YVEX_TEST_ASSERT(summary.template_exists == 1, "template exists");
    YVEX_TEST_ASSERT(summary.imatrix_exists == 1, "imatrix exists");
    YVEX_TEST_ASSERT(summary.output_exists == 0, "ready output optional");

    memset(&summary, 0, sizeof(summary));
    rc = yvex_quant_job_validate("build/tests/quant-job/job.json", &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "validate ready manifest");
    YVEX_TEST_ASSERT_STREQ(summary.name, "test-job", "validated name");
    YVEX_TEST_ASSERT(summary.output_exists == 0, "validated output optional");
    return 0;
}

static int test_missing_tool_reported(void)
{
    yvex_quant_job_options options;
    yvex_quant_job_summary summary;
    yvex_error err;
    int rc;

    fill_options(&options);
    options.tool_path = "build/tests/quant-job/missing-tool";
    options.status = YVEX_QUANT_JOB_STATUS_RUNNING;
    yvex_error_clear(&err);
    rc = yvex_quant_job_write_json("build/tests/quant-job/missing-tool.json", &options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write missing tool manifest");
    YVEX_TEST_ASSERT(summary.tool_exists == 0, "missing tool reported");
    rc = yvex_quant_job_validate("build/tests/quant-job/missing-tool.json", &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "running missing output allowed");
    YVEX_TEST_ASSERT(summary.tool_exists == 0, "validated missing tool reported");
    return 0;
}

static int test_succeeded_requires_output(void)
{
    yvex_quant_job_options options;
    yvex_quant_job_summary summary;
    yvex_error err;
    int rc;

    fill_options(&options);
    options.status = YVEX_QUANT_JOB_STATUS_SUCCEEDED;
    yvex_error_clear(&err);
    rc = yvex_quant_job_write_json("build/tests/quant-job/succeeded.json", &options, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "write succeeded manifest");
    rc = yvex_quant_job_validate("build/tests/quant-job/succeeded.json", &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "succeeded without output fails");

    YVEX_TEST_ASSERT(write_file("build/tests/quant-job/out.gguf", "fake\n"), "write output");
    rc = yvex_quant_job_validate("build/tests/quant-job/succeeded.json", &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "succeeded with output validates");
    YVEX_TEST_ASSERT(summary.output_exists == 1, "output exists");
    return 0;
}

static int test_invalid_args(void)
{
    yvex_quant_job_options options;
    yvex_quant_job_summary summary;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    yvex_error_clear(&err);
    rc = yvex_quant_job_write_json("build/tests/quant-job/invalid.json", &options, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "invalid args fail");
    rc = yvex_quant_job_validate(NULL, &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "invalid validate fails");
    return 0;
}

int yvex_test_quant_job(void)
{
    if (test_names() != 0) return 1;
    if (test_write_validate_ready() != 0) return 1;
    if (test_missing_tool_reported() != 0) return 1;
    if (test_succeeded_requires_output() != 0) return 1;
    if (test_invalid_args() != 0) return 1;
    return 0;
}
