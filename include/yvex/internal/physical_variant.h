/* Bounded serialization contract for immutable physical-variant plans. */
#ifndef INCLUDE_YVEX_INTERNAL_PHYSICAL_VARIANT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_PHYSICAL_VARIANT_H_INCLUDED

#include <yvex/artifact.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_PHYSICAL_VARIANT_FILE_SCHEMA_VERSION 1u

struct yvex_quant_plan;
typedef struct {
    unsigned int file_schema_version, plan_schema_version;
    char profile_name[64];
    char profile_identity[YVEX_SHA256_HEX_CAP];
    char policy_identity[YVEX_SHA256_HEX_CAP];
    char imatrix_identity[YVEX_SHA256_HEX_CAP];
    char physical_variant_identity[YVEX_SHA256_HEX_CAP];
    char payload_plan_identity[YVEX_SHA256_HEX_CAP];
    char transform_identity[YVEX_SHA256_HEX_CAP];
    char required_payload_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long source_snapshot_identity, mapping_identity;
    unsigned long long decision_count, encoded_bytes;
    int complete;
} yvex_quant_plan_file_summary;

int yvex_quant_plan_file_write(const char *path,
                               const struct yvex_quant_plan *plan,
                               yvex_error *err);
int yvex_quant_plan_file_validate(const char *path,
                                  const struct yvex_quant_plan *plan,
                                  yvex_error *err);
/* Rebinding may retain immutable creation identities while requiring the exact
 * current physical decisions, policy, payload recipe, and source mapping. */
int yvex_quant_plan_file_validate_physical_equivalence(
    const char *path, const struct yvex_quant_plan *plan,
    yvex_quant_plan_file_summary *creation, yvex_error *err);
/* Probe authenticates the canonical header and accounts for every decision. */
int yvex_quant_plan_file_probe(const char *path,
                               yvex_quant_plan_file_summary *summary,
                               yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_PHYSICAL_VARIANT_H_INCLUDED */
