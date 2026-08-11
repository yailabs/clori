/*
 * Own one immutable-at-consumption policy document and its template projection.
 *
 * Every rule owns its selector and derives support from canonical qtype capabilities. Policy
 * selects admissible representations; it neither converts bytes nor writes GGUF.
 */
#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/io.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/model.h>
#include <yvex/qtype.h>
#include <yvex/quant.h>

struct yvex_quant_policy {
    unsigned int schema_version;
    char *name;
    char *architecture;
    char *preset_name;
    char *source_kind;
    char *template_path;
    yvex_quant_policy_rule *rules;
    unsigned long long rule_count;
    unsigned long long rule_cap;
    yvex_quant_policy_summary summary;
};

typedef struct {
    int value;
    const char *name;
} policy_name;

typedef struct {
    yvex_quant_qtype qtype;
    yvex_dtype dtype;
} qtype_dtype;

static const policy_name selector_names[] = {
    {YVEX_QUANT_SELECTOR_UNKNOWN, "unknown"},
    {YVEX_QUANT_SELECTOR_ROLE, "role"},
    {YVEX_QUANT_SELECTOR_TENSOR_NAME, "tensor_name"},
    {YVEX_QUANT_SELECTOR_TENSOR_PATTERN, "pattern"},
    {YVEX_QUANT_SELECTOR_LAYER_RANGE, "layer_range"},
    {YVEX_QUANT_SELECTOR_EXPERT_GROUP, "expert_group"},
    {YVEX_QUANT_SELECTOR_DEFAULT, "default"},
};
static const policy_name selector_aliases[] = {
    {YVEX_QUANT_SELECTOR_TENSOR_NAME, "name"},
    {YVEX_QUANT_SELECTOR_TENSOR_PATTERN, "tensor_pattern"},
};
static const char *const policy_status_names[] = {
    "quant-policy-unknown", "quant-policy-valid", "quant-policy-partial", "quant-policy-invalid",
};
static const char *const policy_issue_names[] = {
    "none", "unknown_qtype", "unsupported_storage_qtype", "unsupported_compute_qtype",
    "unknown_role", "unmatched_selector", "template_qtype_mismatch", "requires_imatrix", "format",
};
static const qtype_dtype qtype_dtypes[] = {
    {YVEX_QUANT_QTYPE_F32, YVEX_DTYPE_F32},
    {YVEX_QUANT_QTYPE_F16, YVEX_DTYPE_F16},
    {YVEX_QUANT_QTYPE_BF16, YVEX_DTYPE_BF16},
    {YVEX_QUANT_QTYPE_Q8_0, YVEX_DTYPE_Q8_0},
    {YVEX_QUANT_QTYPE_Q4_0, YVEX_DTYPE_Q4_0},
    {YVEX_QUANT_QTYPE_Q4_K, YVEX_DTYPE_Q4_K},
    {YVEX_QUANT_QTYPE_Q5_K, YVEX_DTYPE_Q5_K},
    {YVEX_QUANT_QTYPE_Q6_K, YVEX_DTYPE_Q6_K},
    {YVEX_QUANT_QTYPE_Q2_K, YVEX_DTYPE_Q2_K},
    {YVEX_QUANT_QTYPE_IQ2_XXS, YVEX_DTYPE_IQ2_XXS},
    {YVEX_QUANT_QTYPE_IQ2_XS, YVEX_DTYPE_IQ2_XS},
    {YVEX_QUANT_QTYPE_IQ3_XXS, YVEX_DTYPE_IQ3_XXS},
    {YVEX_QUANT_QTYPE_IQ4_NL, YVEX_DTYPE_IQ4_NL},
    {YVEX_QUANT_QTYPE_I32, YVEX_DTYPE_I32},
};

static const char *const operation_names[] = {
    "any", "identity", "decode_scale_pair", "checked_cast", "reshape", "transpose",
    "concatenate", "stack", "aggregate", "expert_aggregate",
};
static const char *const physical_class_names[] = {"any", "exact", "quantizable"};
static const char *const collection_names[] = {
    "global", "attention", "compressor", "indexer", "norm", "mhc", "router",
    "routed_expert", "shared_expert", "auxiliary",
};
static const char *const scope_names[] = {"global", "main_layer", "draft"};

static int policy_name_value(const policy_name *rows, size_t count, const char *name,
                             int fallback) {
    size_t index;
    if (name)
        for (index = 0u; index < count; ++index)
            if (strcmp(rows[index].name, name) == 0)
                return rows[index].value;
    return fallback;
}

static int policy_add_rule(yvex_quant_policy *policy, yvex_quant_selector_kind selector_kind,
                           const char *selector, yvex_tensor_role role, yvex_quant_qtype qtype,
                           int requires_imatrix, yvex_error *err);
static int policy_add_rule_v2(yvex_quant_policy *policy,
                              const yvex_quant_policy_rule *source,
                              yvex_error *err);

static int policy_parse_json(yvex_quant_policy **out, const char *path, yvex_error *err);

int yvex_quant_policy_validate(yvex_quant_policy *policy, const char *template_path,
                               yvex_error *err);

static int policy_write_json_file(const char *out_path, const yvex_quant_policy *policy,
                                  yvex_error *err);

static yvex_quant_qtype qtype_from_name(const char *name);
static yvex_quant_selector_kind selector_from_name(const char *name);
static yvex_tensor_role role_from_name(const char *name);
static yvex_dtype qtype_to_dtype(yvex_quant_qtype qtype);
static int qtype_storage_supported(yvex_quant_qtype qtype);
static int qtype_compute_supported(yvex_quant_qtype qtype);

static int policy_json_bool(yvex_gguf_json *json, int *out) {
    return yvex_json_bool(&json->cursor, out) ? YVEX_OK
                                              : yvex_gguf_json_fail(json, "expected boolean");
}

const char *yvex_quant_qtype_name(yvex_quant_qtype qtype) {
    yvex_dtype dtype;

    if (qtype == YVEX_QUANT_QTYPE_SOURCE)
        return "SOURCE";
    if (qtype == YVEX_QUANT_QTYPE_OTHER)
        return "OTHER";
    if (qtype == YVEX_QUANT_QTYPE_MXFP4)
        return "MXFP4";
    dtype = qtype_to_dtype(qtype);
    return dtype == YVEX_DTYPE_UNKNOWN ? "UNKNOWN" : yvex_dtype_name(dtype);
}

static yvex_quant_qtype qtype_from_name(const char *name) {
    yvex_quant_qtype qtype;

    if (!name)
        return YVEX_QUANT_QTYPE_UNKNOWN;
    if (strcmp(name, "OTHER") == 0)
        return YVEX_QUANT_QTYPE_OTHER;
    if (strcmp(name, "SOURCE") == 0)
        return YVEX_QUANT_QTYPE_SOURCE;
    if (strcmp(name, "MXFP4") == 0)
        return YVEX_QUANT_QTYPE_MXFP4;
    for (qtype = YVEX_QUANT_QTYPE_F32; qtype < YVEX_QUANT_QTYPE_OTHER;
         qtype = (yvex_quant_qtype)(qtype + 1)) {
        if (strcmp(name, yvex_quant_qtype_name(qtype)) == 0)
            return qtype;
    }
    return YVEX_QUANT_QTYPE_UNKNOWN;
}

/*
 * Render one policy operation selector without importing internal IR declarations.
 *
 * Returns borrowed process-lifetime text.
 */
const char *yvex_quant_policy_operation_name(yvex_quant_policy_operation operation) {
    return operation <= YVEX_QUANT_POLICY_OPERATION_EXPERT_AGGREGATE
               ? operation_names[operation]
               : "unknown";
}

/*
 * Render one source/terminal physical-class selector.
 *
 * Returns borrowed process-lifetime text.
 */
const char *yvex_quant_policy_physical_class_name(
    yvex_quant_policy_physical_class physical_class) {
    return physical_class <= YVEX_QUANT_POLICY_PHYSICAL_QUANTIZABLE
               ? physical_class_names[physical_class]
               : "unknown";
}

static int policy_table_index(const char *const *names, size_t count, const char *name,
                              int fallback) {
    size_t index;
    if (name)
        for (index = 0u; index < count; ++index)
            if (strcmp(names[index], name) == 0)
                return (int)index;
    return fallback;
}

static yvex_quant_policy_operation operation_from_name(const char *name) {
    return (yvex_quant_policy_operation)policy_table_index(
        operation_names, sizeof(operation_names) / sizeof(operation_names[0]), name,
        YVEX_QUANT_POLICY_OPERATION_ANY);
}

static yvex_quant_policy_physical_class physical_class_from_name(const char *name) {
    return (yvex_quant_policy_physical_class)policy_table_index(
        physical_class_names, sizeof(physical_class_names) / sizeof(physical_class_names[0]), name,
        YVEX_QUANT_POLICY_PHYSICAL_ANY);
}

static yvex_tensor_collection collection_from_name(const char *name) {
    return (yvex_tensor_collection)policy_table_index(
        collection_names, sizeof(collection_names) / sizeof(collection_names[0]), name,
        YVEX_TENSOR_COLLECTION_COUNT);
}

static yvex_tensor_scope scope_from_name(const char *name) {
    return (yvex_tensor_scope)policy_table_index(
        scope_names, sizeof(scope_names) / sizeof(scope_names[0]), name,
        YVEX_TENSOR_SCOPE_DRAFT + 1);
}

const char *yvex_quant_selector_kind_name(yvex_quant_selector_kind kind) {
    return kind >= YVEX_QUANT_SELECTOR_UNKNOWN && kind <= YVEX_QUANT_SELECTOR_DEFAULT
               ? selector_names[kind].name
               : selector_names[YVEX_QUANT_SELECTOR_UNKNOWN].name;
}

static yvex_quant_selector_kind selector_from_name(const char *name) {
    int value = policy_name_value(selector_names, sizeof(selector_names) / sizeof(selector_names[0]),
                                  name, YVEX_QUANT_SELECTOR_UNKNOWN);
    return (yvex_quant_selector_kind)(value != YVEX_QUANT_SELECTOR_UNKNOWN
                                          ? value
                                          : policy_name_value(selector_aliases,
                                                              sizeof(selector_aliases) /
                                                                  sizeof(selector_aliases[0]),
                                                              name, YVEX_QUANT_SELECTOR_UNKNOWN));
}

const char *yvex_quant_policy_status_name(yvex_quant_policy_status status) {
    return status >= YVEX_QUANT_POLICY_STATUS_UNKNOWN && status <= YVEX_QUANT_POLICY_STATUS_INVALID
               ? policy_status_names[status]
               : policy_status_names[YVEX_QUANT_POLICY_STATUS_UNKNOWN];
}

const char *yvex_quant_policy_issue_kind_name(yvex_quant_policy_issue_kind issue) {
    return issue >= YVEX_QUANT_POLICY_ISSUE_NONE && issue <= YVEX_QUANT_POLICY_ISSUE_FORMAT
               ? policy_issue_names[issue]
               : policy_issue_names[YVEX_QUANT_POLICY_ISSUE_FORMAT];
}

static yvex_dtype qtype_to_dtype(yvex_quant_qtype qtype) {
    size_t index;
    for (index = 0u; index < sizeof(qtype_dtypes) / sizeof(qtype_dtypes[0]); ++index)
        if (qtype_dtypes[index].qtype == qtype)
            return qtype_dtypes[index].dtype;
    return YVEX_DTYPE_UNKNOWN;
}

static int qtype_storage_supported(yvex_quant_qtype qtype) {
    const yvex_quant_numeric_capability *capability =
        yvex_quant_numeric_capability_by_name(yvex_quant_qtype_name(qtype));
    return qtype == YVEX_QUANT_QTYPE_SOURCE || (capability && capability->storage_admitted);
}

static int qtype_compute_supported(yvex_quant_qtype qtype) {
    const yvex_quant_numeric_capability *capability =
        yvex_quant_numeric_capability_by_name(yvex_quant_qtype_name(qtype));
    return qtype == YVEX_QUANT_QTYPE_SOURCE ||
           (capability && capability->dedicated_cpu_compute_available);
}

static yvex_tensor_role role_from_name(const char *name) {
    unsigned int i;

    if (!name)
        return YVEX_TENSOR_ROLE_UNKNOWN;
    for (i = 0; i < (unsigned int)YVEX_TENSOR_ROLE_COUNT; ++i) {
        yvex_tensor_role role = (yvex_tensor_role)i;
        if (strcmp(name, yvex_tensor_role_name(role)) == 0)
            return role;
    }
    return YVEX_TENSOR_ROLE_UNKNOWN;
}

/*
 * Hash every policy-v2 field without native structure representation.
 *
 * Writes the canonical policy identity into its summary. Excludes pointers, paths, padding,
 * allocation order, and process state.
 */
static int policy_identity_compute(yvex_quant_policy *policy) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    if (!policy || !policy->name || !policy->architecture)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.quant_policy.identity.v2") ||
        !yvex_sha256_update_u64(&hash, policy->schema_version) ||
        !yvex_sha256_update_text(&hash, policy->name) ||
        !yvex_sha256_update_text(&hash, policy->architecture) ||
        !yvex_sha256_update_text(&hash, policy->preset_name ? policy->preset_name : "custom") ||
        !yvex_sha256_update_u64(&hash, policy->rule_count))
        return 0;
    for (index = 0u; index < policy->rule_count; ++index) {
        const yvex_quant_policy_rule *rule = &policy->rules[index];
        if (!yvex_sha256_update_u64(&hash, rule->schema_version) ||
            !yvex_sha256_update_u64(&hash, rule->match_mask) ||
            !yvex_sha256_update_u64(&hash, rule->role) ||
            !yvex_sha256_update_u64(&hash, rule->collection) ||
            !yvex_sha256_update_u64(&hash, rule->scope) ||
            !yvex_sha256_update_text(&hash, rule->tensor_name ? rule->tensor_name : "") ||
            !yvex_sha256_update_text(&hash, rule->tensor_pattern ? rule->tensor_pattern : "") ||
            !yvex_sha256_update_u64(&hash, rule->layer_first) ||
            !yvex_sha256_update_u64(&hash, rule->layer_last) ||
            !yvex_sha256_update_u64(&hash, rule->expert_group) ||
            !yvex_sha256_update_u64(&hash, rule->operation) ||
            !yvex_sha256_update_u64(&hash, rule->physical_class) ||
            !yvex_sha256_update_u64(&hash, rule->qtype) ||
            !yvex_sha256_update_u64(&hash, (unsigned int)rule->requires_imatrix) ||
            !yvex_sha256_update_u64(&hash, (unsigned int)rule->requires_cpu_compute) ||
            !yvex_sha256_update_u64(&hash, (unsigned int)rule->requires_cuda_compute) ||
            !yvex_sha256_update_u64(&hash, rule->priority) ||
            !yvex_sha256_update_text(&hash, rule->label ? rule->label : ""))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, policy->summary.policy_identity);
    return 1;
}

static void qp_refresh_summary(yvex_quant_policy *policy) {
    unsigned long long i;

    memset(&policy->summary, 0, sizeof(policy->summary));
    policy->summary.schema_version = policy->schema_version;
    policy->summary.name = policy->name;
    policy->summary.architecture = policy->architecture;
    policy->summary.preset_name = policy->preset_name;
    policy->summary.rule_count = policy->rule_count;
    policy->summary.status =
        policy->rule_count > 0 ? YVEX_QUANT_POLICY_STATUS_VALID : YVEX_QUANT_POLICY_STATUS_INVALID;
    for (i = 0; i < policy->rule_count; ++i) {
        yvex_quant_policy_rule *rule = &policy->rules[i];
        if (rule->requires_imatrix)
            policy->summary.requires_imatrix_count++;
        if (rule->storage_supported)
            policy->summary.storage_supported_count++;
        if (rule->compute_supported)
            policy->summary.compute_supported_count++;
        if (rule->qtype == YVEX_QUANT_QTYPE_UNKNOWN ||
            rule->selector_kind == YVEX_QUANT_SELECTOR_UNKNOWN ||
            (rule->selector_kind == YVEX_QUANT_SELECTOR_ROLE &&
             rule->role == YVEX_TENSOR_ROLE_UNKNOWN)) {
            policy->summary.issue_count++;
            policy->summary.status = YVEX_QUANT_POLICY_STATUS_INVALID;
        } else if (!rule->storage_supported || !rule->compute_supported) {
            policy->summary.issue_count++;
            if (policy->summary.status == YVEX_QUANT_POLICY_STATUS_VALID) {
                policy->summary.status = YVEX_QUANT_POLICY_STATUS_PARTIAL;
            }
        }
    }
    (void)policy_identity_compute(policy);
}

static int policy_add_rule(yvex_quant_policy *policy, yvex_quant_selector_kind selector_kind,
                           const char *selector, yvex_tensor_role role, yvex_quant_qtype qtype,
                           int requires_imatrix, yvex_error *err) {
    yvex_quant_policy_rule rule;

    if (!policy || !selector) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_add",
                       "policy and selector are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(&rule, 0, sizeof(rule));
    rule.schema_version = policy->schema_version;
    rule.selector_kind = selector_kind;
    rule.selector = selector;
    rule.role = role;
    rule.qtype = qtype;
    rule.requires_imatrix = requires_imatrix ? 1 : 0;
    rule.requires_cpu_compute = 1;
    rule.requires_cuda_compute = 0;
    rule.priority = 0u;
    if (selector_kind == YVEX_QUANT_SELECTOR_ROLE)
        rule.match_mask = YVEX_QUANT_MATCH_ROLE;
    else if (selector_kind == YVEX_QUANT_SELECTOR_TENSOR_NAME) {
        rule.match_mask = YVEX_QUANT_MATCH_TENSOR_NAME;
        rule.tensor_name = selector;
    } else if (selector_kind == YVEX_QUANT_SELECTOR_TENSOR_PATTERN) {
        rule.match_mask = YVEX_QUANT_MATCH_TENSOR_PATTERN;
        rule.tensor_pattern = selector;
    } else if (selector_kind == YVEX_QUANT_SELECTOR_DEFAULT)
        rule.match_mask = YVEX_QUANT_MATCH_DEFAULT;
    return policy_add_rule_v2(policy, &rule, err);
}

/*
 * Append one fully typed conjunctive rule with independent owned strings.
 *
 * Grows bounded rule storage and copies every executable string. Malformed rule, overflow, or
 * allocation failure leaves prior rules valid. Append does not seal identity or resolve any
 * tensor.
 */
static int policy_add_rule_v2(yvex_quant_policy *policy,
                              const yvex_quant_policy_rule *source,
                              yvex_error *err) {
    yvex_quant_policy_rule *next;
    yvex_quant_policy_rule *rule;
    unsigned long long cap;

    if (!policy || !source || !source->match_mask) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_add_v2",
                       "policy and a nonempty typed matcher are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (policy->rule_count == policy->rule_cap) {
        cap = policy->rule_cap == 0 ? 8u : policy->rule_cap * 2u;
        if (cap < policy->rule_cap || cap > SIZE_MAX / sizeof(policy->rules[0])) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "quant_policy_add_v2", "rule capacity overflowed");
            return YVEX_ERR_BOUNDS;
        }
        next = (yvex_quant_policy_rule *)realloc(policy->rules,
                                                 (size_t)cap * sizeof(policy->rules[0]));
        if (!next) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "quant_policy_add_v2", "rule allocation failed");
            return YVEX_ERR_NOMEM;
        }
        policy->rules = next;
        policy->rule_cap = cap;
    }
    rule = &policy->rules[policy->rule_count];
    *rule = *source;
    rule->schema_version = policy->schema_version;
    rule->selector = source->selector ? yvex_core_strdup(source->selector) : NULL;
    rule->tensor_name = source->tensor_name ? yvex_core_strdup(source->tensor_name) : NULL;
    rule->tensor_pattern = source->tensor_pattern ? yvex_core_strdup(source->tensor_pattern) : NULL;
    rule->label = source->label ? yvex_core_strdup(source->label) : NULL;
    if ((source->selector && !rule->selector) || (source->tensor_name && !rule->tensor_name) ||
        (source->tensor_pattern && !rule->tensor_pattern) || (source->label && !rule->label)) {
        free((char *)rule->selector);
        free((char *)rule->tensor_name);
        free((char *)rule->tensor_pattern);
        free((char *)rule->label);
        memset(rule, 0, sizeof(*rule));
        yvex_error_set(err, YVEX_ERR_NOMEM, "quant_policy_add_v2", "rule string allocation failed");
        return YVEX_ERR_NOMEM;
    }
    rule->storage_supported = qtype_storage_supported(rule->qtype);
    rule->compute_supported = qtype_compute_supported(rule->qtype);
    policy->rule_count++;
    qp_refresh_summary(policy);
    return YVEX_OK;
}

int yvex_quant_policy_open(yvex_quant_policy **out, const char *path, yvex_error *err) {
    int rc = policy_parse_json(out, path, err);
    if (rc == YVEX_OK) {
        rc = yvex_quant_policy_validate(*out, NULL, err);
    }
    return rc;
}

void yvex_quant_policy_close(yvex_quant_policy *policy) {
    unsigned long long i;

    if (!policy)
        return;
    free(policy->name);
    free(policy->architecture);
    free(policy->preset_name);
    free(policy->source_kind);
    free(policy->template_path);
    for (i = 0; i < policy->rule_count; ++i) {
        free((char *)policy->rules[i].selector);
        free((char *)policy->rules[i].tensor_name);
        free((char *)policy->rules[i].tensor_pattern);
        free((char *)policy->rules[i].label);
    }
    free(policy->rules);
    free(policy);
}

/* Serialize one policy through the canonical deterministic JSON writer. */
int yvex_quant_policy_write_json(const char *out_path, const yvex_quant_policy *policy,
                                 yvex_error *err) {
    return policy_write_json_file(out_path, policy, err);
}

int yvex_quant_policy_get_summary(const yvex_quant_policy *policy, yvex_quant_policy_summary *out,
                                  yvex_error *err) {
    if (!policy || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_summary",
                       "policy and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = policy->summary;
    return YVEX_OK;
}

/*
 * Rederive and compare the complete policy identity after admission.
 *
 * Absent, malformed, or stale identity returns typed refusal.
 */
int yvex_quant_policy_identity_validate(const yvex_quant_policy *policy, yvex_error *err) {
    yvex_quant_policy copy;
    char expected[YVEX_QUANT_POLICY_IDENTITY_CAP];

    if (!policy || !yvex_sha256_hex_is_valid(policy->summary.policy_identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_identity",
                       "sealed policy identity is required");
        return YVEX_ERR_INVALID_ARG;
    }
    copy = *policy;
    memcpy(expected, policy->summary.policy_identity, sizeof(expected));
    memset(copy.summary.policy_identity, 0, sizeof(copy.summary.policy_identity));
    if (!policy_identity_compute(&copy) || strcmp(expected, copy.summary.policy_identity) != 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "quant_policy_identity",
                       "policy identity does not match canonical fields");
        return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_quant_policy_create_definition(
    yvex_quant_policy **out, const yvex_quant_policy_definition *definition, yvex_error *err) {
    yvex_quant_policy *policy;
    unsigned long long ordinal;
    int rc = YVEX_OK;

    if (!out || !definition || !definition->name || !definition->architecture ||
        !definition->source_kind || !definition->rules || !definition->rule_count) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_definition",
                       "complete policy definition is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    policy = (yvex_quant_policy *)calloc(1, sizeof(*policy));
    if (!policy) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "quant_policy_definition",
                       "policy allocation failed");
        return YVEX_ERR_NOMEM;
    }
    policy->schema_version = YVEX_QUANT_POLICY_SCHEMA_VERSION;
    policy->name = yvex_core_strdup(definition->name);
    policy->architecture = yvex_core_strdup(definition->architecture);
    policy->preset_name = yvex_core_strdup(definition->name);
    policy->source_kind = yvex_core_strdup(definition->source_kind);
    if (!policy->name || !policy->architecture || !policy->preset_name || !policy->source_kind) {
        rc = YVEX_ERR_NOMEM;
        yvex_error_set(err, rc, "quant_policy_definition", "policy string allocation failed");
        goto done;
    }
    for (ordinal = 0u; rc == YVEX_OK && ordinal < definition->rule_count; ++ordinal)
        rc = policy_add_rule_v2(policy, &definition->rules[ordinal], err);
    if (rc == YVEX_OK)
        rc = yvex_quant_policy_validate(policy, NULL, err);
    if (rc == YVEX_OK) {
        *out = policy;
        policy = NULL;
    }

done:
    yvex_quant_policy_close(policy);
    return rc;
}

unsigned long long yvex_quant_policy_rule_count(const yvex_quant_policy *policy) {
    return policy ? policy->rule_count : 0;
}

const yvex_quant_policy_rule *yvex_quant_policy_rule_at(const yvex_quant_policy *policy,
                                                        unsigned long long index) {
    if (!policy || index >= policy->rule_count)
        return NULL;
    return &policy->rules[index];
}

static yvex_quant_qtype template_qtype_from_dtype(yvex_dtype dtype) {
    size_t index;
    for (index = 0u; index < sizeof(qtype_dtypes) / sizeof(qtype_dtypes[0]); ++index)
        if (qtype_dtypes[index].dtype == dtype)
            return qtype_dtypes[index].qtype;
    return YVEX_QUANT_QTYPE_OTHER;
}

static int qp_has_role_qtype(const yvex_quant_policy *policy, yvex_tensor_role role,
                             yvex_quant_qtype qtype) {
    unsigned long long i;

    for (i = 0; i < policy->rule_count; ++i) {
        const yvex_quant_policy_rule *rule = &policy->rules[i];
        if (rule->selector_kind == YVEX_QUANT_SELECTOR_ROLE && rule->role == role &&
            rule->qtype == qtype) {
            return 1;
        }
    }
    return 0;
}

int yvex_quant_policy_create_from_template(yvex_quant_policy **out, const char *template_path,
                                           const char *architecture, yvex_error *err) {
    yvex_artifact_options artifact_options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_quant_policy *policy = NULL;
    unsigned long long i;
    int rc;

    if (!out || !template_path || !architecture) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_derive",
                       "out, template_path, and architecture are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    policy = (yvex_quant_policy *)calloc(1, sizeof(*policy));
    if (!policy) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "quant_policy_derive", "policy allocation failed");
        return YVEX_ERR_NOMEM;
    }
    policy->schema_version = 1u;
    policy->name = yvex_core_strdup("template-derived-policy");
    policy->architecture = yvex_core_strdup(architecture);
    policy->source_kind = yvex_core_strdup("template-derived");
    policy->template_path = yvex_core_strdup(template_path);
    if (!policy->name || !policy->architecture || !policy->source_kind || !policy->template_path) {
        rc = YVEX_ERR_NOMEM;
        yvex_error_set(err, YVEX_ERR_NOMEM, "quant_policy_derive",
                       "policy string allocation failed");
        goto done;
    }

    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = template_path;
    artifact_options.readonly = 1;
    artifact_options.map = 1;
    rc = yvex_artifact_open(&artifact, &artifact_options, err);
    if (rc == YVEX_OK)
        rc = yvex_gguf_open(&gguf, artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&tensors, gguf, err);
    if (rc != YVEX_OK)
        goto done;

    for (i = 0; i < yvex_tensor_table_count(tensors); ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, i);
        yvex_quant_qtype qtype;

        if (!tensor)
            continue;
        qtype = template_qtype_from_dtype(tensor->dtype);
        if (tensor->role != YVEX_TENSOR_ROLE_UNKNOWN) {
            if (qp_has_role_qtype(policy, tensor->role, qtype))
                continue;
            rc = policy_add_rule(policy, YVEX_QUANT_SELECTOR_ROLE,
                                 yvex_tensor_role_name(tensor->role), tensor->role, qtype, 0, err);
        } else {
            rc = policy_add_rule(policy, YVEX_QUANT_SELECTOR_TENSOR_NAME, tensor->name,
                                 YVEX_TENSOR_ROLE_UNKNOWN, qtype, 0, err);
        }
        if (rc != YVEX_OK)
            goto done;
    }
    rc = yvex_quant_policy_validate(policy, NULL, err);
    if (rc != YVEX_OK)
        goto done;
    *out = policy;
    policy = NULL;
    rc = YVEX_OK;

done:
    yvex_quant_policy_close(policy);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc;
}

static int qj_parse_source(yvex_gguf_json *j, yvex_quant_policy *policy) {
    int rc = yvex_gguf_json_expect(j, '{');
    if (rc != YVEX_OK)
        return rc;
    while (j->cursor.cursor < j->cursor.end) {
        char *key = NULL;
        int complete = 0;

        rc = yvex_gguf_json_member(j, &key, &complete);
        if (rc != YVEX_OK) return rc;
        if (complete) return YVEX_OK;
        if (strcmp(key, "kind") == 0) {
            free(policy->source_kind);
            policy->source_kind = yvex_gguf_json_string(j);
        } else if (strcmp(key, "template_path") == 0) {
            free(policy->template_path);
            policy->template_path = yvex_gguf_json_string(j);
        } else {
            rc = yvex_gguf_json_skip(j);
        }
        free(key);
        if (rc != YVEX_OK)
            return rc;
        yvex_gguf_json_optional_comma(j);
    }
    return yvex_gguf_json_fail(j, "unterminated source object");
}

static int policy_json_u64(yvex_gguf_json *json, unsigned long long *out) {
    return yvex_json_u64(&json->cursor, out) ? YVEX_OK
                                             : yvex_gguf_json_fail(json, "expected unsigned integer");
}

static int policy_parse_match(yvex_gguf_json *json, yvex_quant_policy_rule *rule) {
    int rc = yvex_gguf_json_expect(json, '{');
    int have_first = 0;
    int have_last = 0;
    int complete = 0;

    while (rc == YVEX_OK && json->cursor.cursor < json->cursor.end) {
        char *key = NULL;
        int member_complete = 0;
        rc = yvex_gguf_json_member(json, &key, &member_complete);
        if (rc != YVEX_OK || member_complete) {
            complete = member_complete;
            free(key);
            break;
        }
        if (strcmp(key, "role") == 0) {
            char *value = yvex_gguf_json_string(json);
            rule->role = role_from_name(value);
            rule->match_mask |= YVEX_QUANT_MATCH_ROLE;
            if (!value || rule->role == YVEX_TENSOR_ROLE_UNKNOWN)
                rc = yvex_gguf_json_fail(json, "unknown role matcher");
            free(value);
        } else if (strcmp(key, "collection") == 0) {
            char *value = yvex_gguf_json_string(json);
            rule->collection = collection_from_name(value);
            rule->match_mask |= YVEX_QUANT_MATCH_COLLECTION;
            if (!value || rule->collection >= YVEX_TENSOR_COLLECTION_COUNT)
                rc = yvex_gguf_json_fail(json, "unknown collection matcher");
            free(value);
        } else if (strcmp(key, "scope") == 0) {
            char *value = yvex_gguf_json_string(json);
            rule->scope = scope_from_name(value);
            rule->match_mask |= YVEX_QUANT_MATCH_SCOPE;
            if (!value || rule->scope > YVEX_TENSOR_SCOPE_DRAFT)
                rc = yvex_gguf_json_fail(json, "unknown scope matcher");
            free(value);
        } else if (strcmp(key, "tensor_name") == 0) {
            free((char *)rule->tensor_name);
            rule->tensor_name = yvex_gguf_json_string(json);
            rule->match_mask |= YVEX_QUANT_MATCH_TENSOR_NAME;
            if (!rule->tensor_name) rc = yvex_error_code(json->err);
        } else if (strcmp(key, "tensor_pattern") == 0) {
            free((char *)rule->tensor_pattern);
            rule->tensor_pattern = yvex_gguf_json_string(json);
            rule->match_mask |= YVEX_QUANT_MATCH_TENSOR_PATTERN;
            if (!rule->tensor_pattern) rc = yvex_error_code(json->err);
        } else if (strcmp(key, "layer_first") == 0) {
            rc = policy_json_u64(json, &rule->layer_first);
            rule->match_mask |= YVEX_QUANT_MATCH_LAYER_RANGE;
            have_first = rc == YVEX_OK;
        } else if (strcmp(key, "layer_last") == 0) {
            rc = policy_json_u64(json, &rule->layer_last);
            rule->match_mask |= YVEX_QUANT_MATCH_LAYER_RANGE;
            have_last = rc == YVEX_OK;
        } else if (strcmp(key, "expert_group") == 0) {
            rc = policy_json_u64(json, &rule->expert_group);
            rule->match_mask |= YVEX_QUANT_MATCH_EXPERT_GROUP;
        } else if (strcmp(key, "operation") == 0) {
            char *value = yvex_gguf_json_string(json);
            rule->operation = operation_from_name(value);
            rule->match_mask |= YVEX_QUANT_MATCH_OPERATION;
            if (!value || (rule->operation == YVEX_QUANT_POLICY_OPERATION_ANY &&
                           strcmp(value, "any") != 0))
                rc = yvex_gguf_json_fail(json, "unknown operation matcher");
            free(value);
        } else if (strcmp(key, "physical_class") == 0) {
            char *value = yvex_gguf_json_string(json);
            rule->physical_class = physical_class_from_name(value);
            rule->match_mask |= YVEX_QUANT_MATCH_PHYSICAL_CLASS;
            if (!value || (rule->physical_class == YVEX_QUANT_POLICY_PHYSICAL_ANY &&
                           strcmp(value, "any") != 0))
                rc = yvex_gguf_json_fail(json, "unknown physical class matcher");
            free(value);
        } else if (strcmp(key, "default") == 0) {
            int value = 0;
            rc = policy_json_bool(json, &value);
            if (rc == YVEX_OK && !value)
                rc = yvex_gguf_json_fail(json, "default matcher must be true");
            rule->match_mask |= YVEX_QUANT_MATCH_DEFAULT;
        } else {
            rc = yvex_gguf_json_fail(json, "unknown executable policy matcher");
        }
        free(key);
        if (rc == YVEX_OK) yvex_gguf_json_optional_comma(json);
    }
    if (rc == YVEX_OK && !complete)
        rc = yvex_gguf_json_fail(json, "unterminated policy matcher");
    if (rc == YVEX_OK && (have_first != have_last ||
                          (have_first && rule->layer_first > rule->layer_last)))
        rc = yvex_gguf_json_fail(json, "layer matcher requires an ordered closed range");
    if (rc == YVEX_OK && !rule->match_mask)
        rc = yvex_gguf_json_fail(json, "empty policy matcher is forbidden");
    return rc;
}

static int policy_parse_action(yvex_gguf_json *json, yvex_quant_policy_rule *rule,
                               int *have_qtype) {
    int rc = yvex_gguf_json_expect(json, '{');
    int complete = 0;

    while (rc == YVEX_OK && json->cursor.cursor < json->cursor.end) {
        char *key = NULL;
        int member_complete = 0;
        rc = yvex_gguf_json_member(json, &key, &member_complete);
        if (rc != YVEX_OK || member_complete) {
            complete = member_complete;
            free(key);
            break;
        }
        if (strcmp(key, "qtype") == 0) {
            char *value = yvex_gguf_json_string(json);
            rule->qtype = qtype_from_name(value);
            *have_qtype = value && rule->qtype != YVEX_QUANT_QTYPE_UNKNOWN;
            if (!*have_qtype) rc = yvex_gguf_json_fail(json, "unknown action qtype");
            free(value);
        } else if (strcmp(key, "calibration") == 0) {
            char *value = yvex_gguf_json_string(json);
            if (!value || (strcmp(value, "none") != 0 && strcmp(value, "optional") != 0 &&
                           strcmp(value, "required") != 0))
                rc = yvex_gguf_json_fail(json, "unknown calibration action");
            else
                rule->requires_imatrix = strcmp(value, "required") == 0;
            free(value);
        } else if (strcmp(key, "requires_imatrix") == 0) {
            rc = policy_json_bool(json, &rule->requires_imatrix);
        } else if (strcmp(key, "requires_cpu_compute") == 0) {
            rc = policy_json_bool(json, &rule->requires_cpu_compute);
        } else if (strcmp(key, "requires_cuda_compute") == 0) {
            rc = policy_json_bool(json, &rule->requires_cuda_compute);
        } else {
            rc = yvex_gguf_json_fail(json, "unknown executable policy action");
        }
        free(key);
        if (rc == YVEX_OK) yvex_gguf_json_optional_comma(json);
    }
    if (rc == YVEX_OK && !complete)
        rc = yvex_gguf_json_fail(json, "unterminated policy action");
    return rc;
}

static int qj_parse_rule(yvex_gguf_json *j, void *context) {
    yvex_quant_policy *policy = context;
    yvex_quant_policy_rule rule;
    char *selector_kind = NULL;
    char *selector = NULL;
    char *qtype = NULL;
    char *label = NULL;
    int requires_imatrix = 0;
    int have_match = 0;
    int have_action = 0;
    int have_qtype = 0;
    yvex_quant_selector_kind kind;
    yvex_quant_qtype qt;
    yvex_tensor_role role = YVEX_TENSOR_ROLE_UNKNOWN;
    int rc = yvex_gguf_json_expect(j, '{');

    memset(&rule, 0, sizeof(rule));
    rule.schema_version = policy->schema_version;
    rule.requires_cpu_compute = 1;
    rule.requires_cuda_compute = 1;
    if (rc != YVEX_OK)
        return rc;
    while (j->cursor.cursor < j->cursor.end) {
        char *key = NULL;
        int complete = 0;

        rc = yvex_gguf_json_member(j, &key, &complete);
        if (rc != YVEX_OK) goto done;
        if (complete) break;
        if (strcmp(key, "selector_kind") == 0) {
            free(selector_kind);
            selector_kind = yvex_gguf_json_string(j);
            if (!selector_kind)
                rc = yvex_error_code(j->err);
        } else if (strcmp(key, "selector") == 0) {
            free(selector);
            selector = yvex_gguf_json_string(j);
            if (!selector)
                rc = yvex_error_code(j->err);
        } else if (strcmp(key, "qtype") == 0) {
            free(qtype);
            qtype = yvex_gguf_json_string(j);
            if (!qtype)
                rc = yvex_error_code(j->err);
        } else if (strcmp(key, "requires_imatrix") == 0) {
            rc = policy_json_bool(j, &requires_imatrix);
        } else if (strcmp(key, "match") == 0) {
            rc = policy_parse_match(j, &rule);
            have_match = rc == YVEX_OK;
        } else if (strcmp(key, "action") == 0) {
            rc = policy_parse_action(j, &rule, &have_qtype);
            have_action = rc == YVEX_OK;
        } else if (strcmp(key, "priority") == 0) {
            unsigned long long priority = 0u;
            rc = policy_json_u64(j, &priority);
            if (rc == YVEX_OK && priority > UINT_MAX)
                rc = yvex_gguf_json_fail(j, "policy priority exceeds U32");
            rule.priority = (unsigned int)priority;
        } else if (strcmp(key, "label") == 0) {
            free(label);
            label = yvex_gguf_json_string(j);
            if (!label) rc = yvex_error_code(j->err);
        } else {
            rc = yvex_gguf_json_fail(j, "unknown executable policy rule field");
        }
        free(key);
        if (rc != YVEX_OK)
            goto done;
        yvex_gguf_json_optional_comma(j);
    }
    if (have_match || have_action) {
        if (!have_match || !have_action || !have_qtype) {
            rc = yvex_gguf_json_fail(j, "policy-v2 rule requires match and action objects");
            goto done;
        }
        rule.label = label;
        rc = policy_add_rule_v2(policy, &rule, j->err);
    } else {
        if (!selector_kind || !selector || !qtype) {
            rc = yvex_gguf_json_fail(j, "policy rule missing selector_kind, selector, or qtype");
            goto done;
        }
        kind = selector_from_name(selector_kind);
        qt = qtype_from_name(qtype);
        if (kind == YVEX_QUANT_SELECTOR_ROLE)
            role = role_from_name(selector);
        rc = policy_add_rule(policy, kind, selector, role, qt, requires_imatrix, j->err);
    }

done:
    free(selector_kind);
    free(selector);
    free(qtype);
    free(label);
    free((char *)rule.tensor_name);
    free((char *)rule.tensor_pattern);
    return rc;
}

static int policy_parse_json(yvex_quant_policy **out, const char *path, yvex_error *err) {
    yvex_quant_policy *policy;
    yvex_gguf_json j;
    int rc;

    if (!out || !path) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_json", "out and path are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    rc = yvex_gguf_json_open(&j, path, "quant_policy_json", err);
    if (rc != YVEX_OK)
        return rc;
    policy = (yvex_quant_policy *)calloc(1, sizeof(*policy));
    if (!policy) {
        yvex_gguf_json_close(&j);
        yvex_error_set(err, YVEX_ERR_NOMEM, "quant_policy_json", "policy allocation failed");
        return YVEX_ERR_NOMEM;
    }
    policy->schema_version = 1u;
    rc = yvex_gguf_json_expect(&j, '{');
    if (rc != YVEX_OK)
        goto fail;
    while (j.cursor.cursor < j.cursor.end) {
        char *key = NULL;
        int complete = 0;

        rc = yvex_gguf_json_member(&j, &key, &complete);
        if (rc != YVEX_OK) goto fail;
        if (complete) break;
        if (strcmp(key, "schema") == 0) {
            char *schema = yvex_gguf_json_string(&j);
            if (!schema) {
                free(key);
                rc = yvex_error_code(err);
                goto fail;
            }
            if (strcmp(schema, "yvex.quant_policy.v1") == 0)
                policy->schema_version = 1u;
            else if (strcmp(schema, "yvex.quant_policy.v2") == 0)
                policy->schema_version = YVEX_QUANT_POLICY_SCHEMA_VERSION;
            else {
                free(schema);
                free(key);
                rc = yvex_gguf_json_fail(&j, "unsupported quant policy schema");
                goto fail;
            }
            free(schema);
        } else if (strcmp(key, "name") == 0) {
            free(policy->name);
            policy->name = yvex_gguf_json_string(&j);
            if (!policy->name)
                rc = yvex_error_code(err);
        } else if (strcmp(key, "architecture") == 0) {
            free(policy->architecture);
            policy->architecture = yvex_gguf_json_string(&j);
            if (!policy->architecture)
                rc = yvex_error_code(err);
        } else if (strcmp(key, "source") == 0) {
            rc = qj_parse_source(&j, policy);
        } else if (strcmp(key, "preset") == 0) {
            free(policy->preset_name);
            policy->preset_name = yvex_gguf_json_string(&j);
            if (!policy->preset_name)
                rc = yvex_error_code(err);
        } else if (strcmp(key, "rules") == 0) {
            rc = yvex_gguf_json_array(&j, qj_parse_rule, policy,
                                      "malformed rules array", "unterminated rules array");
        } else if (policy->schema_version == 1u) {
            rc = yvex_gguf_json_skip(&j);
        } else {
            rc = yvex_gguf_json_fail(&j, "unknown policy-v2 root field");
        }
        free(key);
        if (rc != YVEX_OK)
            goto fail;
        yvex_gguf_json_optional_comma(&j);
    }
    if (!policy->name)
        policy->name = yvex_core_strdup("unnamed-policy");
    if (!policy->architecture)
        policy->architecture = yvex_core_strdup("unknown");
    if (!policy->name || !policy->architecture) {
        rc = YVEX_ERR_NOMEM;
        yvex_error_set(err, YVEX_ERR_NOMEM, "quant_policy_json", "policy string allocation failed");
        goto fail;
    }
    *out = policy;
    yvex_gguf_json_close(&j);
    return YVEX_OK;

fail:
    yvex_quant_policy_close(policy);
    yvex_gguf_json_close(&j);
    return rc;
}

static int policy_write_json_file(const char *out_path, const yvex_quant_policy *policy,
                                  yvex_error *err) {
    FILE *fp;
    unsigned long long i;

    if (!out_path || !policy) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_json",
                       "out_path and policy are required");
        return YVEX_ERR_INVALID_ARG;
    }
    fp = fopen(out_path, "wb");
    if (!fp) {
        yvex_error_setf(err, YVEX_ERR_IO, "quant_policy_json", "cannot open output policy: %s",
                        out_path);
        return YVEX_ERR_IO;
    }
    fprintf(fp, "{\n");
    fprintf(fp, "  \"schema\": \"yvex.quant_policy.v%u\",\n", policy->schema_version);
    fprintf(fp, "  \"name\": ");
    yvex_file_json_write_string(fp, policy->name);
    fprintf(fp, ",\n  \"architecture\": ");
    yvex_file_json_write_string(fp, policy->architecture);
    fprintf(fp, ",\n");
    if (policy->preset_name) {
        fprintf(fp, "  \"preset\": ");
        yvex_file_json_write_string(fp, policy->preset_name);
        fprintf(fp, ",\n");
    }
    if (policy->source_kind || policy->template_path) {
        fprintf(fp, "  \"source\": {\n");
        fprintf(fp, "    \"kind\": ");
        yvex_file_json_write_string(fp,
                                 policy->source_kind ? policy->source_kind : "template-derived");
        fprintf(fp, ",\n    \"template_path\": ");
        yvex_file_json_write_string(fp, policy->template_path);
        fprintf(fp, "\n  },\n");
    }
    fprintf(fp, "  \"rules\": [\n");
    for (i = 0; i < policy->rule_count; ++i) {
        const yvex_quant_policy_rule *rule = &policy->rules[i];
        if (policy->schema_version == 1u) {
            fprintf(fp, "    {\"selector_kind\": ");
            yvex_file_json_write_string(fp, yvex_quant_selector_kind_name(rule->selector_kind));
            fprintf(fp, ", \"selector\": ");
            yvex_file_json_write_string(fp, rule->selector);
            fprintf(fp, ", \"qtype\": ");
            yvex_file_json_write_string(fp, yvex_quant_qtype_name(rule->qtype));
            fprintf(fp, ", \"requires_imatrix\": %s}",
                    rule->requires_imatrix ? "true" : "false");
        } else {
            fprintf(fp, "    {\"match\": {");
            {
                int comma = 0;
#define WRITE_MATCH_TEXT(bit, key, value)                                                         \
    do {                                                                                           \
        if (rule->match_mask & (bit)) {                                                            \
            fprintf(fp, "%s\"%s\": ", comma ? ", " : "", (key));                            \
            yvex_file_json_write_string(fp, (value));                                              \
            comma = 1;                                                                             \
        }                                                                                          \
    } while (0)
                if (rule->match_mask & YVEX_QUANT_MATCH_ROLE)
                    WRITE_MATCH_TEXT(YVEX_QUANT_MATCH_ROLE, "role",
                                     yvex_tensor_role_name(rule->role));
                if (rule->match_mask & YVEX_QUANT_MATCH_COLLECTION)
                    WRITE_MATCH_TEXT(YVEX_QUANT_MATCH_COLLECTION, "collection",
                                     collection_names[rule->collection]);
                if (rule->match_mask & YVEX_QUANT_MATCH_SCOPE)
                    WRITE_MATCH_TEXT(YVEX_QUANT_MATCH_SCOPE, "scope", scope_names[rule->scope]);
                WRITE_MATCH_TEXT(YVEX_QUANT_MATCH_TENSOR_NAME, "tensor_name", rule->tensor_name);
                WRITE_MATCH_TEXT(YVEX_QUANT_MATCH_TENSOR_PATTERN, "tensor_pattern",
                                 rule->tensor_pattern);
                if (rule->match_mask & YVEX_QUANT_MATCH_LAYER_RANGE) {
                    fprintf(fp, "%s\"layer_first\": %llu, \"layer_last\": %llu",
                            comma ? ", " : "", rule->layer_first, rule->layer_last);
                    comma = 1;
                }
                if (rule->match_mask & YVEX_QUANT_MATCH_EXPERT_GROUP) {
                    fprintf(fp, "%s\"expert_group\": %llu", comma ? ", " : "",
                            rule->expert_group);
                    comma = 1;
                }
                if (rule->match_mask & YVEX_QUANT_MATCH_OPERATION)
                    WRITE_MATCH_TEXT(YVEX_QUANT_MATCH_OPERATION, "operation",
                                     yvex_quant_policy_operation_name(rule->operation));
                if (rule->match_mask & YVEX_QUANT_MATCH_PHYSICAL_CLASS)
                    WRITE_MATCH_TEXT(YVEX_QUANT_MATCH_PHYSICAL_CLASS, "physical_class",
                                     yvex_quant_policy_physical_class_name(rule->physical_class));
                if (rule->match_mask & YVEX_QUANT_MATCH_DEFAULT)
                    fprintf(fp, "%s\"default\": true", comma ? ", " : "");
#undef WRITE_MATCH_TEXT
            }
            fprintf(fp, "}, \"action\": {\"qtype\": ");
            yvex_file_json_write_string(fp, yvex_quant_qtype_name(rule->qtype));
            fprintf(fp,
                    ", \"calibration\": \"%s\", \"requires_cpu_compute\": %s, "
                    "\"requires_cuda_compute\": %s}, \"priority\": %u",
                    rule->requires_imatrix ? "required" : "none",
                    rule->requires_cpu_compute ? "true" : "false",
                    rule->requires_cuda_compute ? "true" : "false", rule->priority);
            if (rule->label) {
                fprintf(fp, ", \"label\": ");
                yvex_file_json_write_string(fp, rule->label);
            }
            fprintf(fp, "}");
        }
        fprintf(fp, "%s\n", i + 1u == policy->rule_count ? "" : ",");
    }
    fprintf(fp, "  ]\n}\n");
    if (fclose(fp) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "quant_policy_json", "failed closing output policy: %s",
                        out_path);
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}

static yvex_quant_qtype validate_qtype_from_dtype(yvex_dtype dtype) {
    return template_qtype_from_dtype(dtype);
}

static void qp_set_summary(yvex_quant_policy *policy, unsigned long long extra_issues, int fatal) {
    unsigned long long i;

    memset(&policy->summary, 0, sizeof(policy->summary));
    policy->summary.schema_version = policy->schema_version;
    policy->summary.name = policy->name;
    policy->summary.architecture = policy->architecture;
    policy->summary.preset_name = policy->preset_name;
    policy->summary.rule_count = policy->rule_count;
    policy->summary.status =
        policy->rule_count > 0 ? YVEX_QUANT_POLICY_STATUS_VALID : YVEX_QUANT_POLICY_STATUS_INVALID;
    policy->summary.issue_count = extra_issues;
    if (extra_issues > 0) {
        policy->summary.status =
            fatal ? YVEX_QUANT_POLICY_STATUS_INVALID : YVEX_QUANT_POLICY_STATUS_PARTIAL;
    }

    for (i = 0; i < policy->rule_count; ++i) {
        yvex_quant_policy_rule *rule = &policy->rules[i];
        rule->storage_supported = qtype_storage_supported(rule->qtype);
        rule->compute_supported = qtype_compute_supported(rule->qtype);
        if (rule->requires_imatrix) {
            policy->summary.requires_imatrix_count++;
        }
        if (rule->storage_supported)
            policy->summary.storage_supported_count++;
        else {
            policy->summary.issue_count++;
            if (policy->summary.status == YVEX_QUANT_POLICY_STATUS_VALID) {
                policy->summary.status = YVEX_QUANT_POLICY_STATUS_PARTIAL;
            }
        }
        if (rule->compute_supported)
            policy->summary.compute_supported_count++;
        else {
            policy->summary.issue_count++;
            if (policy->summary.status == YVEX_QUANT_POLICY_STATUS_VALID) {
                policy->summary.status = YVEX_QUANT_POLICY_STATUS_PARTIAL;
            }
        }
        if (rule->qtype == YVEX_QUANT_QTYPE_UNKNOWN ||
            (policy->schema_version == 1u &&
             (rule->selector_kind == YVEX_QUANT_SELECTOR_UNKNOWN ||
              (rule->selector_kind == YVEX_QUANT_SELECTOR_ROLE &&
               rule->role == YVEX_TENSOR_ROLE_UNKNOWN))) ||
            (policy->schema_version == YVEX_QUANT_POLICY_SCHEMA_VERSION && !rule->match_mask)) {
            policy->summary.issue_count++;
            policy->summary.status = YVEX_QUANT_POLICY_STATUS_INVALID;
        }
    }
    (void)policy_identity_compute(policy);
}

static int qp_match_pattern(const char *pattern, const char *name) {
    const char *star;
    size_t prefix_len;
    size_t suffix_len;
    size_t name_len;

    if (!pattern || !name)
        return 0;
    if (strcmp(pattern, "*") == 0)
        return 1;
    star = strchr(pattern, '*');
    if (!star)
        return strcmp(pattern, name) == 0;
    prefix_len = (size_t)(star - pattern);
    suffix_len = strlen(star + 1);
    name_len = strlen(name);
    if (name_len < prefix_len + suffix_len)
        return 0;
    if (strncmp(pattern, name, prefix_len) != 0)
        return 0;
    if (suffix_len > 0 && strcmp(name + name_len - suffix_len, star + 1) != 0)
        return 0;
    return 1;
}

static int qp_validate_template(yvex_quant_policy *policy, const char *template_path,
                                unsigned long long *issues, yvex_error *err) {
    yvex_artifact_options artifact_options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    unsigned long long i;
    int rc;

    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = template_path;
    artifact_options.readonly = 1;
    artifact_options.map = 1;
    rc = yvex_artifact_open(&artifact, &artifact_options, err);
    if (rc == YVEX_OK)
        rc = yvex_gguf_open(&gguf, artifact, err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&tensors, gguf, err);
    if (rc != YVEX_OK)
        goto done;

    for (i = 0; i < policy->rule_count; ++i) {
        const yvex_quant_policy_rule *rule = &policy->rules[i];
        unsigned long long j;
        int matched = 0;

        for (j = 0; j < yvex_tensor_table_count(tensors); ++j) {
            const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, j);
            int applies = 0;

            if (!tensor)
                continue;
            if (rule->selector_kind == YVEX_QUANT_SELECTOR_DEFAULT)
                applies = 1;
            else if (rule->selector_kind == YVEX_QUANT_SELECTOR_ROLE && tensor->role == rule->role)
                applies = 1;
            else if (rule->selector_kind == YVEX_QUANT_SELECTOR_TENSOR_NAME &&
                     strcmp(rule->selector, tensor->name) == 0)
                applies = 1;
            else if (rule->selector_kind == YVEX_QUANT_SELECTOR_TENSOR_PATTERN &&
                     qp_match_pattern(rule->selector, tensor->name))
                applies = 1;
            if (!applies)
                continue;
            matched = 1;
            if (validate_qtype_from_dtype(tensor->dtype) != rule->qtype) {
                (*issues)++;
            }
        }
        if (!matched)
            (*issues)++;
    }

done:
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc;
}

/*
 * Validate policy identity, rules, capabilities, and optional template equivalence.
 *
 * May allocate default identity strings and replaces the embedded summary.
 */
int yvex_quant_policy_validate(yvex_quant_policy *policy, const char *template_path,
                               yvex_error *err) {
    unsigned long long template_issues = 0;
    int rc = YVEX_OK;

    if (!policy) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "quant_policy_validate", "policy is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (policy->schema_version != 1u &&
        policy->schema_version != YVEX_QUANT_POLICY_SCHEMA_VERSION) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "quant_policy_validate",
                       "unsupported quant policy schema");
        return YVEX_ERR_UNSUPPORTED;
    }
    if (!policy->name)
        policy->name = yvex_core_strdup("unnamed-policy");
    if (!policy->architecture)
        policy->architecture = yvex_core_strdup("unknown");
    if (!policy->name || !policy->architecture) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "quant_policy_validate",
                       "policy string allocation failed");
        return YVEX_ERR_NOMEM;
    }
    if (template_path) {
        rc = qp_validate_template(policy, template_path, &template_issues, err);
        if (rc != YVEX_OK)
            return rc;
    }
    qp_set_summary(policy, template_issues, policy->rule_count == 0);
    return YVEX_OK;
}
