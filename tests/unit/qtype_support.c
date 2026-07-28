/*
 * YVEX - qtype support tests
 */
#include <string.h>

#include <yvex/api.h>
#include <yvex/internal/quant_numeric.h>

#include "tests/test.h"

int yvex_test_qtype_support(void)
{
    const yvex_qtype_support_info *row;
    const yvex_quant_numeric_capability *capability;

    YVEX_TEST_ASSERT(yvex_qtype_support_count() == 43u,
                     "support projection covers every pinned qtype identity");
    row = yvex_qtype_support_by_name("F32");
    YVEX_TEST_ASSERT(row && row->policy_supported &&
                     yvex_qtype_support_storage_supported(row) && row->emit_supported,
                     "F32 row");
    YVEX_TEST_ASSERT_STREQ(yvex_qtype_support_name(row), "F32", "F32 projected name");
    row = yvex_qtype_support_by_name("F16");
    YVEX_TEST_ASSERT(row && row->emit_supported && row->quantize_supported, "F16 row");
    row = yvex_qtype_support_by_name("Q8_0");
    YVEX_TEST_ASSERT(row && row->policy_supported &&
                     yvex_qtype_support_storage_supported(row) &&
                     row->emit_supported && row->quantize_supported &&
                     row->compute_supported,
                     "Q8_0 row");
    row = yvex_qtype_support_by_name("Q2_K");
    YVEX_TEST_ASSERT(row && row->policy_supported &&
                     yvex_qtype_support_storage_supported(row) &&
                     row->emit_supported && row->quantize_supported &&
                     row->compute_supported,
                     "Q2_K row");
    row = yvex_qtype_support_by_name("Q4_2");
    YVEX_TEST_ASSERT(row && !row->policy_supported && !row->emit_supported,
                     "removed qtype projects deterministic refusal");
    capability = yvex_quant_numeric_capability_by_name("Q8_0");
    YVEX_TEST_ASSERT(capability && capability->dedicated_cpu_compute_available,
                     "canonical Q8_0 row admits dedicated CPU compute");
    capability = yvex_quant_numeric_capability_by_name("Q2_K");
    YVEX_TEST_ASSERT(capability && capability->dedicated_cuda_compute_available,
                     "canonical Q2_K row admits dedicated CUDA compute");
    row = yvex_qtype_support_by_name("IQ2_XXS");
    YVEX_TEST_ASSERT(row && row->policy_supported && row->emit_supported &&
                         row->quantize_supported && row->compute_supported,
                     "IQ2_XXS projects its calibrated full vertical");
    capability = yvex_quant_numeric_capability_by_name("IQ2_XXS");
    YVEX_TEST_ASSERT(capability && capability->calibration ==
                                           YVEX_QUANT_CALIBRATION_REQUIRED &&
                         capability->dedicated_cpu_compute_available &&
                         capability->dedicated_cuda_compute_available,
                     "IQ2_XXS requires calibration and admits CPU/CUDA compute");
    capability = yvex_quant_numeric_capability_by_name("Q4_K");
    YVEX_TEST_ASSERT(capability && !capability->dedicated_cpu_compute_available,
                     "canonical Q4_K row preserves CPU compute refusal");
    YVEX_TEST_ASSERT(yvex_qtype_support_by_name("NOPE") == NULL, "unknown missing");
    return 0;
}
