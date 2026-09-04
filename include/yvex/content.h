/* Ordered provider content and exact model input/output capability facts. */
#ifndef YVEX_CONTENT_H
#define YVEX_CONTENT_H

#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_CONTENT_PART_SCHEMA_V1 1u
#define YVEX_MODEL_CAPABILITY_SCHEMA_V1 1u
#define YVEX_CONTENT_MAX_PARTS 32u
#define YVEX_CONTENT_MEDIA_TYPE_CAP 96u
#define YVEX_CONTENT_REFERENCE_CAP 1024u
#define YVEX_CONTENT_ID_CAP 65u
#define YVEX_CONTENT_WIRE_MAX_BYTES 1572864u
#define YVEX_CONTENT_LOCAL_MAX_BYTES 68719476736ull

typedef enum {
    YVEX_CONTENT_TEXT = 0,
    YVEX_CONTENT_IMAGE,
    YVEX_CONTENT_AUDIO,
    YVEX_CONTENT_VIDEO,
    YVEX_CONTENT_FILE,
    YVEX_CONTENT_TENSOR,
    YVEX_CONTENT_KIND_COUNT
} yvex_content_kind;

typedef enum {
    YVEX_CONTENT_INLINE = 0,
    YVEX_CONTENT_LOCAL_FILE
} yvex_content_storage;

typedef struct {
    unsigned int schema_version;
    yvex_content_kind kind;
    yvex_content_storage storage;
    const unsigned char *bytes;
    unsigned long long byte_count;
    unsigned long long width, height, duration_milliseconds;
    char media_type[YVEX_CONTENT_MEDIA_TYPE_CAP];
    char reference[YVEX_CONTENT_REFERENCE_CAP];
    char content_identity[YVEX_CONTENT_ID_CAP];
    char derived_from_content_identity[YVEX_CONTENT_ID_CAP];
} yvex_content_part;

#define YVEX_CONTENT_KIND_MASK(kind) (1ull << (unsigned int)(kind))
#define YVEX_MODEL_CAPABILITY_ORDERED_INPUT_PARTS (1ull << 0u)
#define YVEX_MODEL_CAPABILITY_STATEFUL_SESSION (1ull << 1u)
#define YVEX_MODEL_CAPABILITY_STREAMING_OUTPUT (1ull << 2u)
#define YVEX_MODEL_CAPABILITY_DEMAND_ACTIVATION (1ull << 3u)

typedef struct {
    unsigned int schema_version;
    unsigned long long input_kinds, output_kinds;
    unsigned long long execution_properties;
    unsigned long long maximum_input_parts;
} yvex_model_capability_summary;

typedef enum {
    YVEX_MODEL_CAPABILITY_PROFILE_TEXT_GENERATION = 0,
    YVEX_MODEL_CAPABILITY_PROFILE_CONDITIONED_AUDIOVISUAL_GENERATION
} yvex_model_capability_profile;

int yvex_content_part_seal(yvex_content_part *part, yvex_error *err);
int yvex_content_part_validate(const yvex_content_part *part, yvex_error *err);
int yvex_content_part_local_verify(const yvex_content_part *part,
                                   yvex_error *err);
int yvex_content_parts_validate(const yvex_content_part *parts,
                                unsigned long long count, yvex_error *err);
int yvex_content_parts_identity(
    const yvex_content_part *parts, unsigned long long count,
    char identity[YVEX_CONTENT_ID_CAP], yvex_error *err);
int yvex_content_parts_wire_encode(const yvex_content_part *parts,
                                   unsigned long long count,
                                   unsigned char *output,
                                   unsigned long long capacity,
                                   unsigned long long *byte_count,
                                   yvex_error *err);
int yvex_content_parts_wire_decode(const unsigned char *input,
                                   unsigned long long byte_count,
                                   yvex_content_part **parts,
                                   unsigned long long *count,
                                   yvex_error *err);
void yvex_content_parts_close(yvex_content_part **parts,
                              unsigned long long count);
int yvex_model_capability_validate(
    const yvex_model_capability_summary *capability, yvex_error *err);
int yvex_model_capability_admit(
    const yvex_model_capability_summary *capability,
    const yvex_content_part *parts, unsigned long long count,
    yvex_error *err);
int yvex_model_capability_profile_describe(
    yvex_model_capability_profile profile,
    yvex_model_capability_summary *capability, yvex_error *err);
const char *yvex_content_kind_name(yvex_content_kind kind);
int yvex_model_capability_wire_encode(
    const yvex_model_capability_summary *capability,
    unsigned char output[40], yvex_error *err);
int yvex_model_capability_wire_decode(
    const unsigned char input[40], unsigned long long byte_count,
    yvex_model_capability_summary *capability, yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* YVEX_CONTENT_H */
