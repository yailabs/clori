
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <yvex/api.h>

#include "tests/test.h"

static int mkdir_ok(const char *p) { return mkdir(p, 0777) == 0 || errno == EEXIST; }

static int write_st(const char *path)
{
    const char *json = "{\"model.embed_tokens.weight\":{\"dtype\":\"F16\",\"shape\":[8,4],\"data_offsets\":[0,64]},\"unknown.weight\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[64,66]}}";
    unsigned long long len = (unsigned long long)strlen(json);
    unsigned char b[8];
    FILE *fp = fopen(path, "wb");
    unsigned long long i;
    if (!fp) return 0;
    for (i = 0; i < 8; ++i) b[i] = (unsigned char)((len >> (i * 8u)) & 0xffu);
    fwrite(b, 1, 8, fp);
    fwrite(json, 1, (size_t)len, fp);
    for (i = 0; i < 66; ++i) fputc((int)(i & 0xffu), fp);
    return fclose(fp) == 0;
}

int yvex_test_conversion_plan(void)
{
    yvex_conversion_options options;
    yvex_conversion_summary summary;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/conversion-plan") == 0,
                     "clear conversion plan fixture");
    YVEX_TEST_ASSERT(mkdir_ok("build") && mkdir_ok("build/tests") && mkdir_ok("build/tests/conversion-plan"), "mkdir");
    YVEX_TEST_ASSERT(write_st("build/tests/conversion-plan/model.safetensors"), "write st");
    memset(&options, 0, sizeof(options));
    options.architecture = "qwen3";
    options.native_source_dir = "build/tests/conversion-plan";
    yvex_error_clear(&err);
    rc = yvex_conversion_plan_write_json(&options, "build/tests/conversion-plan/plan.json", &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "plan writes");
    YVEX_TEST_ASSERT(summary.native_tensor_count == 2, "native count");
    YVEX_TEST_ASSERT(summary.planned_tensor_count == 1, "planned count");
    YVEX_TEST_ASSERT(summary.unmapped_tensor_count == 1, "unmapped count");

    rc = yvex_conversion_plan_write_json(NULL, "x", &summary, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "invalid args fail");
    return 0;
}
