
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <yvex/api.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/quant_numeric.h>

#include "tests/test.h"

static int write_policy(const char *path, const char *qtype)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fprintf(fp,
            "{\n"
            "  \"schema\": \"yvex.quant_policy.v1\",\n"
            "  \"name\": \"test-policy\",\n"
            "  \"architecture\": \"deepseek4\",\n"
            "  \"rules\": [\n"
            "    {\"selector_kind\": \"role\", \"selector\": \"token_embedding\", \"qtype\": \"%s\", \"requires_imatrix\": false},\n"
            "    {\"selector_kind\": \"pattern\", \"selector\": \"blk.*.ffn.experts.*\", \"qtype\": \"Q2_K\", \"requires_imatrix\": true}\n"
            "  ]\n"
            "}\n",
            qtype);
    return fclose(fp) == 0;
}

static int test_names(void)
{
    YVEX_TEST_ASSERT_STREQ(yvex_quant_qtype_name(YVEX_QUANT_QTYPE_Q8_0), "Q8_0", "qtype name");
    YVEX_TEST_ASSERT_STREQ(yvex_quant_selector_kind_name(YVEX_QUANT_SELECTOR_ROLE), "role", "selector name");
    YVEX_TEST_ASSERT_STREQ(yvex_quant_policy_status_name(YVEX_QUANT_POLICY_STATUS_PARTIAL), "quant-policy-partial", "status name");
    YVEX_TEST_ASSERT_STREQ(yvex_quant_policy_issue_kind_name(YVEX_QUANT_POLICY_ISSUE_UNKNOWN_QTYPE), "unknown_qtype", "issue name");
    return 0;
}

static int test_open_validate_write(void)
{
    const char *dir = "build/tests/quant-policy";
    const char *path = "build/tests/quant-policy/policy.json";
    const char *out = "build/tests/quant-policy/written.json";
    yvex_quant_policy *policy = NULL;
    yvex_quant_policy_summary summary;
    const yvex_quant_policy_rule *rule;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(system("rm -rf build/tests/quant-policy && mkdir -p build/tests/quant-policy") == 0,
                     "create policy dir");
    (void)dir;
    YVEX_TEST_ASSERT(write_policy(path, "Q8_0"), "write policy");
    yvex_error_clear(&err);
    rc = yvex_quant_policy_open(&policy, path, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open policy");
    YVEX_TEST_ASSERT(yvex_quant_policy_get_summary(policy, &summary, &err) == YVEX_OK, "summary");
    YVEX_TEST_ASSERT(summary.rule_count == 2, "rule count");
    YVEX_TEST_ASSERT(summary.requires_imatrix_count == 1, "requires imatrix count");
    YVEX_TEST_ASSERT(summary.storage_supported_count == 2,
                     "Q8_0 and Q2_K have storage geometry");
    YVEX_TEST_ASSERT(summary.compute_supported_count == 2,
                     "quant policy projects canonical CPU compute truth");
    rule = yvex_quant_policy_rule_at(policy, 0);
    YVEX_TEST_ASSERT(rule && rule->role == YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, "role parsed");
    YVEX_TEST_ASSERT(rule->qtype == YVEX_QUANT_QTYPE_Q8_0, "qtype parsed");
    YVEX_TEST_ASSERT(yvex_quant_policy_write_json(out, policy, &err) == YVEX_OK, "write policy JSON");
    yvex_quant_policy_close(policy);
    return 0;
}

static int test_reject_unknown_qtype(void)
{
    const char *path = "build/tests/quant-policy/bad.json";
    yvex_quant_policy *policy = NULL;
    yvex_quant_policy_summary summary;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(write_policy(path, "BAD_QTYPE"), "write bad policy");
    yvex_error_clear(&err);
    rc = yvex_quant_policy_open(&policy, path, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "unknown qtype opens as invalid policy");
    YVEX_TEST_ASSERT(yvex_quant_policy_get_summary(policy, &summary, &err) == YVEX_OK, "bad summary");
    YVEX_TEST_ASSERT(summary.status == YVEX_QUANT_POLICY_STATUS_INVALID, "unknown qtype invalid");
    yvex_quant_policy_close(policy);
    return 0;
}

static int test_derive_fixture(void)
{
    yvex_quant_policy *policy = NULL;
    yvex_quant_policy_summary summary;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);
    rc = yvex_quant_policy_create_from_template(&policy,
                                                "tests/fixtures/gguf/valid-tokenizer-simple.gguf",
                                                "llama",
                                                &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "derive fixture policy");
    YVEX_TEST_ASSERT(yvex_quant_policy_get_summary(policy, &summary, &err) == YVEX_OK, "derive summary");
    YVEX_TEST_ASSERT(summary.rule_count >= 1, "derived rules");
    yvex_quant_policy_close(policy);
    return 0;
}

static int write_policy_v2(const char *path, unsigned int priority)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fprintf(fp,
            "{\n"
            "  \"schema\": \"yvex.quant_policy.v2\",\n"
            "  \"name\": \"conjunctive-policy\",\n"
            "  \"architecture\": \"deepseek4-v4-flash-dspark\",\n"
            "  \"rules\": [\n"
            "    {\"match\": {\"role\": \"moe_expert_gate\", "
            "\"scope\": \"main_layer\", \"layer_first\": 2, \"layer_last\": 41, "
            "\"operation\": \"expert_aggregate\", \"physical_class\": \"quantizable\"}, "
            "\"action\": {\"qtype\": \"IQ2_XXS\", \"calibration\": \"required\", "
            "\"requires_cpu_compute\": true, \"requires_cuda_compute\": true}, "
            "\"priority\": %u, \"label\": \"calibrated routed gate\"},\n"
            "    {\"match\": {\"default\": true}, "
            "\"action\": {\"qtype\": \"Q8_0\", \"calibration\": \"none\", "
            "\"requires_cpu_compute\": true, \"requires_cuda_compute\": true}, "
            "\"priority\": 1, \"label\": \"default\"}\n"
            "  ]\n"
            "}\n", priority);
    return fclose(fp) == 0;
}

static int test_policy_v2_and_presets(void)
{
    static const char *const preset_identities[] = {
        "84a934800f49b2cb7cf031eb18c8c1660ef8250da751e0ea34f65852d7c5333f",
        "adb3f1549440ad39adb2cc08826e659e91ac95c289b9ce75a0efc3dbf89d60a3",
        "5942d4c1bc6ae5d140c72387ff93da14e5ed9a77a6ded3f41ac3f7e9596a0f24",
    };
    const char *path = "build/tests/quant-policy/v2.json";
    const char *roundtrip = "build/tests/quant-policy/v2-roundtrip.json";
    const char *mutated = "build/tests/quant-policy/v2-mutated.json";
    yvex_quant_policy *policy = NULL;
    yvex_quant_policy *opened = NULL;
    yvex_quant_policy_summary summary;
    yvex_quant_policy_summary roundtrip_summary;
    const yvex_quant_policy_rule *rule;
    unsigned long long preset;
    char original_identity[YVEX_QUANT_POLICY_IDENTITY_CAP];
    yvex_error err;

    YVEX_TEST_ASSERT(write_policy_v2(path, 200u), "write policy v2");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_quant_policy_open(&policy, path, &err) == YVEX_OK,
                     "open policy v2");
    YVEX_TEST_ASSERT(yvex_quant_policy_validate(policy, NULL, &err) == YVEX_OK,
                     "validate policy v2");
    YVEX_TEST_ASSERT(yvex_quant_policy_get_summary(policy, &summary, &err) == YVEX_OK,
                     "policy v2 summary");
    YVEX_TEST_ASSERT(summary.schema_version == YVEX_QUANT_POLICY_SCHEMA_VERSION,
                     "policy v2 schema");
    YVEX_TEST_ASSERT(summary.status == YVEX_QUANT_POLICY_STATUS_VALID,
                     "policy v2 status");
    YVEX_TEST_ASSERT(summary.rule_count == 2u && summary.requires_imatrix_count == 1u,
                     "policy v2 counts");
    rule = yvex_quant_policy_rule_at(policy, 0u);
    YVEX_TEST_ASSERT(rule && rule->priority == 200u, "priority parsed");
    YVEX_TEST_ASSERT((rule->match_mask & YVEX_QUANT_MATCH_ROLE) &&
                         (rule->match_mask & YVEX_QUANT_MATCH_SCOPE) &&
                         (rule->match_mask & YVEX_QUANT_MATCH_LAYER_RANGE) &&
                         (rule->match_mask & YVEX_QUANT_MATCH_OPERATION) &&
                         (rule->match_mask & YVEX_QUANT_MATCH_PHYSICAL_CLASS),
                     "conjunctive matcher parsed");
    YVEX_TEST_ASSERT(rule->qtype == YVEX_QUANT_QTYPE_IQ2_XXS &&
                         rule->requires_imatrix && rule->requires_cpu_compute &&
                         rule->requires_cuda_compute,
                     "policy v2 action parsed");
    memcpy(original_identity, summary.policy_identity, sizeof(original_identity));
    YVEX_TEST_ASSERT(yvex_quant_policy_identity_validate(policy, &err) == YVEX_OK,
                     "policy identity validates");
    YVEX_TEST_ASSERT(yvex_quant_policy_write_json(roundtrip, policy, &err) == YVEX_OK,
                     "write policy v2");
    YVEX_TEST_ASSERT(yvex_quant_policy_open(&opened, roundtrip, &err) == YVEX_OK,
                     "reopen policy v2");
    YVEX_TEST_ASSERT(yvex_quant_policy_get_summary(opened, &roundtrip_summary, &err) == YVEX_OK,
                     "roundtrip policy summary");
    YVEX_TEST_ASSERT_STREQ(roundtrip_summary.policy_identity, original_identity,
                           "policy v2 roundtrip identity");
    yvex_quant_policy_close(opened);
    opened = NULL;
    YVEX_TEST_ASSERT(write_policy_v2(mutated, 201u), "write priority mutation");
    YVEX_TEST_ASSERT(yvex_quant_policy_open(&opened, mutated, &err) == YVEX_OK,
                     "open priority mutation");
    YVEX_TEST_ASSERT(yvex_quant_policy_get_summary(opened, &roundtrip_summary, &err) == YVEX_OK,
                     "mutation summary");
    YVEX_TEST_ASSERT(strcmp(roundtrip_summary.policy_identity, original_identity) != 0,
                     "priority changes policy identity");
    yvex_quant_policy_close(opened);
    yvex_quant_policy_close(policy);

    YVEX_TEST_ASSERT(yvex_quant_policy_preset_count() == 3u, "preset catalog count");
    for (preset = 0u; preset < yvex_quant_policy_preset_count(); ++preset) {
        const char *name = yvex_quant_policy_preset_name(preset);
        policy = NULL;
        YVEX_TEST_ASSERT(name && yvex_quant_policy_preset_open(&policy, name, &err) == YVEX_OK,
                         "open built-in preset");
        YVEX_TEST_ASSERT(yvex_quant_policy_get_summary(policy, &summary, &err) == YVEX_OK &&
                             summary.status == YVEX_QUANT_POLICY_STATUS_VALID &&
                             yvex_quant_policy_identity_validate(policy, &err) == YVEX_OK,
                         "built-in preset is sealed");
        YVEX_TEST_ASSERT_STREQ(summary.policy_identity, preset_identities[preset],
                               "family-owned preset preserves admitted identity");
        yvex_quant_policy_close(policy);
    }
    return 0;
}

static int test_dspark_bootstrap_policy(void)
{
    static const yvex_tensor_role exact_roles[] = {
        YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM,
        YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM,
        YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING,
        YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT,
        YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE,
    };
    yvex_quant_policy *policy = NULL;
    yvex_quant_policy_summary summary;
    yvex_error err;
    unsigned long long index, role_index;
    int have_draft_default = 0;

    YVEX_TEST_ASSERT(strlen(YVEX_DEEPSEEK_QUANT_IMATRIX_SOURCE_IDENTITY) == 64u,
                     "DSpark bootstrap calibration has an explicit source identity");
    YVEX_TEST_ASSERT(YVEX_DEEPSEEK_QUANT_IMATRIX_DATASET_IDENTITY[0] != '\0',
                     "DSpark bootstrap calibration has an explicit dataset identity");

    YVEX_TEST_ASSERT(
        yvex_quant_policy_preset_open(
            &policy, "deepseek-v4-flash-dspark-bootstrap-q2-v1", &err) ==
                YVEX_OK &&
            yvex_quant_policy_get_summary(policy, &summary, &err) == YVEX_OK,
        "open the DSpark bootstrap policy");
    for (index = 0ull; index < summary.rule_count; ++index) {
        const yvex_quant_policy_rule *rule =
            yvex_quant_policy_rule_at(policy, index);
        if (rule &&
            (rule->match_mask & (YVEX_QUANT_MATCH_SCOPE |
                                 YVEX_QUANT_MATCH_PHYSICAL_CLASS)) ==
                (YVEX_QUANT_MATCH_SCOPE | YVEX_QUANT_MATCH_PHYSICAL_CLASS) &&
            rule->scope == YVEX_TENSOR_SCOPE_DRAFT &&
            rule->physical_class == YVEX_QUANT_POLICY_PHYSICAL_QUANTIZABLE &&
            rule->qtype == YVEX_QUANT_QTYPE_Q8_0)
            have_draft_default = 1;
    }
    YVEX_TEST_ASSERT(have_draft_default,
                     "DSpark approximable tensors have an explicit conservative policy");
    for (role_index = 0ull;
         role_index < sizeof(exact_roles) / sizeof(exact_roles[0]); ++role_index) {
        int found = 0;
        for (index = 0ull; index < summary.rule_count; ++index) {
            const yvex_quant_policy_rule *rule =
                yvex_quant_policy_rule_at(policy, index);
            if (rule && rule->role == exact_roles[role_index] &&
                rule->scope == YVEX_TENSOR_SCOPE_DRAFT &&
                rule->qtype == YVEX_QUANT_QTYPE_BF16)
                found = 1;
        }
        YVEX_TEST_ASSERT(found,
                         "DSpark control tensors remain exact in the bootstrap policy");
    }
    yvex_quant_policy_close(policy);
    return 0;
}

int yvex_test_quant_policy(void)
{
    if (test_names() != 0) return 1;
    if (test_open_validate_write() != 0) return 1;
    if (test_reject_unknown_qtype() != 0) return 1;
    if (test_derive_fixture() != 0) return 1;
    if (test_policy_v2_and_presets() != 0) return 1;
    return test_dspark_bootstrap_policy();
}
