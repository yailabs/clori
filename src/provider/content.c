/* Own bounded ordered content identity, local-reference verification, and wire form. */
#define _POSIX_C_SOURCE 200809L
#include <yvex/content.h>
#include <yvex/internal/core.h>

#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    unsigned char *bytes;
    size_t capacity, count;
} content_writer;
typedef struct {
    const unsigned char *bytes;
    size_t count, offset;
} content_reader;

static int content_refuse(yvex_error *err, yvex_status status,
                          const char *reason)
{
    yvex_error_set(err, status, "provider.content", reason);
    return status;
}

static int bounded_text(const char *text, size_t capacity, int required)
{
    size_t count;
    if (!text) return 0;
    count = strnlen(text, capacity);
    return count < capacity && (!required || count != 0u);
}

static int media_type_valid(const char *type)
{
    size_t index, count;
    if (!bounded_text(type, YVEX_CONTENT_MEDIA_TYPE_CAP, 1)) return 0;
    count = strlen(type);
    for (index = 0u; index < count; ++index) {
        unsigned char byte = (unsigned char)type[index];
        if (byte < 0x21u || byte > 0x7eu || byte == '"' || byte == '\\')
            return 0;
    }
    return strchr(type, '/') != NULL;
}

static int part_shape_valid(const yvex_content_part *part)
{
    if (!part || part->schema_version != YVEX_CONTENT_PART_SCHEMA_V1 ||
        part->kind >= YVEX_CONTENT_KIND_COUNT ||
        part->storage > YVEX_CONTENT_LOCAL_FILE ||
        !media_type_valid(part->media_type) ||
        !bounded_text(part->reference, sizeof(part->reference), 0) ||
        part->byte_count > (part->storage == YVEX_CONTENT_INLINE
                                ? YVEX_CONTENT_WIRE_MAX_BYTES
                                : YVEX_CONTENT_LOCAL_MAX_BYTES) ||
        (part->storage == YVEX_CONTENT_INLINE && part->byte_count &&
         !part->bytes))
        return 0;
    if (part->storage == YVEX_CONTENT_INLINE)
        return part->byte_count && !part->reference[0];
    return !part->bytes && part->byte_count && part->reference[0] == '/';
}

static int file_digest(const char *path, unsigned long long expected,
                       char identity[YVEX_CONTENT_ID_CAP], yvex_error *err)
{
    unsigned char buffer[65536], digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    struct stat facts;
    unsigned long long total = 0u;
    ssize_t got;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0 || fstat(fd, &facts) != 0 || !S_ISREG(facts.st_mode) ||
        facts.st_size < 0 || (unsigned long long)facts.st_size != expected) {
        if (fd >= 0) (void)close(fd);
        return content_refuse(err, YVEX_ERR_IO,
                              "local content reference is unavailable or changed");
    }
    yvex_sha256_init(&hash);
    while ((got = read(fd, buffer, sizeof(buffer))) > 0) {
        total += (unsigned long long)got;
        if (total > expected ||
            !yvex_sha256_update(&hash, buffer, (size_t)got)) {
            (void)close(fd);
            return content_refuse(err, YVEX_ERR_BOUNDS,
                                  "local content exceeds its sealed byte extent");
        }
    }
    if (close(fd) != 0 || got < 0 || total != expected ||
        !yvex_sha256_final(&hash, digest))
        return content_refuse(err, YVEX_ERR_IO,
                              "local content could not be read completely");
    yvex_sha256_hex(digest, identity);
    return YVEX_OK;
}

int yvex_content_part_seal(yvex_content_part *part, yvex_error *err)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    int rc;
    if (!part_shape_valid(part))
        return content_refuse(err, YVEX_ERR_INVALID_ARG,
                              "complete bounded content part is required");
    if (part->storage == YVEX_CONTENT_LOCAL_FILE)
        rc = file_digest(part->reference, part->byte_count,
                         part->content_identity, err);
    else {
        yvex_sha256_init(&hash);
        rc = yvex_sha256_update(&hash, part->bytes,
                                (size_t)part->byte_count) &&
                     yvex_sha256_final(&hash, digest)
                 ? YVEX_OK : YVEX_ERR_STATE;
        if (rc == YVEX_OK) yvex_sha256_hex(digest, part->content_identity);
    }
    if (rc != YVEX_OK) return rc;
    if (part->derived_from_content_identity[0] &&
        (!yvex_sha256_hex_valid(part->derived_from_content_identity) ||
         !strcmp(part->derived_from_content_identity,
                 part->content_identity)))
        return content_refuse(err, YVEX_ERR_INVALID_ARG,
                              "derived content requires a distinct source identity");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_content_part_validate(const yvex_content_part *part, yvex_error *err)
{
    if (!part_shape_valid(part) ||
        !yvex_sha256_hex_valid(part->content_identity) ||
        (part->derived_from_content_identity[0] &&
         (!yvex_sha256_hex_valid(part->derived_from_content_identity) ||
          !strcmp(part->derived_from_content_identity,
                  part->content_identity))))
        return content_refuse(err, YVEX_ERR_INVALID_ARG,
                              "sealed content part fields are invalid");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_content_part_local_verify(const yvex_content_part *part,
                                   yvex_error *err)
{
    char identity[YVEX_CONTENT_ID_CAP];
    int rc = yvex_content_part_validate(part, err);
    if (rc != YVEX_OK || part->storage != YVEX_CONTENT_LOCAL_FILE) return rc;
    rc = file_digest(part->reference, part->byte_count, identity, err);
    if (rc == YVEX_OK && strcmp(identity, part->content_identity))
        rc = content_refuse(err, YVEX_ERR_STATE,
                            "local content identity changed after admission");
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_content_parts_validate(const yvex_content_part *parts,
                                unsigned long long count, yvex_error *err)
{
    unsigned long long index;
    if (!parts || !count || count > YVEX_CONTENT_MAX_PARTS)
        return content_refuse(err, YVEX_ERR_INVALID_ARG,
                              "one bounded ordered content collection is required");
    for (index = 0u; index < count; ++index)
        if (yvex_content_part_validate(parts + index, err) != YVEX_OK)
            return yvex_error_code(err);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_content_parts_identity(
    const yvex_content_part *parts, unsigned long long count,
    char identity[YVEX_CONTENT_ID_CAP], yvex_error *err)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long index;
    if (!identity || yvex_content_parts_validate(parts, count, err) != YVEX_OK)
        return yvex_error_code(err);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.ordered-content.v1") ||
        !yvex_sha256_update_u64_be(&hash, count)) goto failed;
    for (index = 0u; index < count; ++index) {
        const yvex_content_part *part = parts + index;
        if (!yvex_sha256_update_u64_be(&hash, index) ||
            !yvex_sha256_update_u64_be(&hash, part->kind) ||
            !yvex_sha256_update_u64_be(&hash, part->storage) ||
            !yvex_sha256_update_u64_be(&hash, part->byte_count) ||
            !yvex_sha256_update_u64_be(&hash, part->width) ||
            !yvex_sha256_update_u64_be(&hash, part->height) ||
            !yvex_sha256_update_u64_be(&hash, part->duration_milliseconds) ||
            !yvex_sha256_update_text(&hash, part->media_type) ||
            !yvex_sha256_update_text(&hash, part->content_identity) ||
            !yvex_sha256_update_text(&hash,
                                     part->derived_from_content_identity))
            goto failed;
    }
    if (!yvex_sha256_final(&hash, digest)) goto failed;
    yvex_sha256_hex(digest, identity);
    yvex_error_clear(err);
    return YVEX_OK;
failed:
    return content_refuse(err, YVEX_ERR_STATE,
                          "ordered content identity derivation failed");
}

static int write_bytes(content_writer *writer, const void *bytes, size_t count)
{
    if (!writer || writer->count > writer->capacity ||
        count > writer->capacity - writer->count) return 0;
    if (count) memcpy(writer->bytes + writer->count, bytes, count);
    writer->count += count;
    return 1;
}

static int write_u64(content_writer *writer, unsigned long long value)
{
    unsigned char bytes[8];
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        bytes[index] = (unsigned char)(value >> ((7u - index) * 8u));
    return write_bytes(writer, bytes, sizeof(bytes));
}

static int write_text(content_writer *writer, const char *value, size_t cap)
{
    size_t count = strnlen(value, cap);
    return count < cap && write_u64(writer, count) &&
           write_bytes(writer, value, count);
}

int yvex_content_parts_wire_encode(const yvex_content_part *parts,
                                   unsigned long long count,
                                   unsigned char *output,
                                   unsigned long long capacity,
                                   unsigned long long *byte_count,
                                   yvex_error *err)
{
    content_writer writer = {output, (size_t)capacity, 0u};
    unsigned long long index;
    if (byte_count) *byte_count = 0u;
    if (!output || !byte_count || capacity > YVEX_CONTENT_WIRE_MAX_BYTES ||
        capacity > SIZE_MAX ||
        yvex_content_parts_validate(parts, count, err) != YVEX_OK)
        return content_refuse(err, YVEX_ERR_INVALID_ARG,
                              "valid content and bounded wire output are required");
    if (!write_u64(&writer, count)) goto bounds;
    for (index = 0u; index < count; ++index) {
        const yvex_content_part *part = parts + index;
        if (!write_u64(&writer, part->schema_version) ||
            !write_u64(&writer, part->kind) ||
            !write_u64(&writer, part->storage) ||
            !write_u64(&writer, part->byte_count) ||
            !write_u64(&writer, part->width) ||
            !write_u64(&writer, part->height) ||
            !write_u64(&writer, part->duration_milliseconds) ||
            !write_text(&writer, part->media_type, sizeof(part->media_type)) ||
            !write_text(&writer, part->reference, sizeof(part->reference)) ||
            !write_text(&writer, part->content_identity,
                        sizeof(part->content_identity)) ||
            !write_text(&writer, part->derived_from_content_identity,
                        sizeof(part->derived_from_content_identity)) ||
            (part->storage == YVEX_CONTENT_INLINE &&
             !write_bytes(&writer, part->bytes, (size_t)part->byte_count)))
            goto bounds;
    }
    *byte_count = writer.count;
    yvex_error_clear(err);
    return YVEX_OK;
bounds:
    return content_refuse(err, YVEX_ERR_BOUNDS,
                          "ordered content exceeds wire capacity");
}

static int read_u64(content_reader *reader, unsigned long long *value)
{
    unsigned int index;
    if (!reader || !value || reader->offset > reader->count ||
        reader->count - reader->offset < 8u) return 0;
    *value = 0u;
    for (index = 0u; index < 8u; ++index)
        *value = (*value << 8u) | reader->bytes[reader->offset++];
    return 1;
}

static int read_text(content_reader *reader, char *value, size_t capacity)
{
    unsigned long long count;
    if (!read_u64(reader, &count) || count >= capacity ||
        count > SIZE_MAX || reader->offset > reader->count ||
        count > reader->count - reader->offset) return 0;
    if (count) memcpy(value, reader->bytes + reader->offset, (size_t)count);
    value[count] = '\0';
    reader->offset += (size_t)count;
    return 1;
}

void yvex_content_parts_close(yvex_content_part **parts,
                              unsigned long long count)
{
    unsigned long long index;
    if (!parts || !*parts) return;
    for (index = 0u; index < count; ++index)
        free((void *)(*parts)[index].bytes);
    free(*parts);
    *parts = NULL;
}

int yvex_content_parts_wire_decode(const unsigned char *input,
                                   unsigned long long byte_count,
                                   yvex_content_part **parts,
                                   unsigned long long *count,
                                   yvex_error *err)
{
    content_reader reader = {input, (size_t)byte_count, 0u};
    yvex_content_part *owned = NULL;
    unsigned long long item_count, index, value;
    if (parts) *parts = NULL;
    if (count) *count = 0u;
    if (!input || !byte_count || byte_count > YVEX_CONTENT_WIRE_MAX_BYTES ||
        byte_count > SIZE_MAX || !parts || !count ||
        !read_u64(&reader, &item_count) || !item_count ||
        item_count > YVEX_CONTENT_MAX_PARTS)
        return content_refuse(err, YVEX_ERR_INVALID_ARG,
                              "bounded content wire bytes and outputs are required");
    owned = calloc((size_t)item_count, sizeof(*owned));
    if (!owned) return content_refuse(err, YVEX_ERR_NOMEM,
                                      "content allocation failed");
    for (index = 0u; index < item_count; ++index) {
        yvex_content_part *part = owned + index;
        if (!read_u64(&reader, &value) || value > UINT_MAX) goto malformed;
        part->schema_version = (unsigned int)value;
        if (!read_u64(&reader, &value) || value >= YVEX_CONTENT_KIND_COUNT)
            goto malformed;
        part->kind = (yvex_content_kind)value;
        if (!read_u64(&reader, &value) || value > YVEX_CONTENT_LOCAL_FILE)
            goto malformed;
        part->storage = (yvex_content_storage)value;
        if (!read_u64(&reader, &part->byte_count) ||
            !read_u64(&reader, &part->width) ||
            !read_u64(&reader, &part->height) ||
            !read_u64(&reader, &part->duration_milliseconds) ||
            !read_text(&reader, part->media_type, sizeof(part->media_type)) ||
            !read_text(&reader, part->reference, sizeof(part->reference)) ||
            !read_text(&reader, part->content_identity,
                       sizeof(part->content_identity)) ||
            !read_text(&reader, part->derived_from_content_identity,
                       sizeof(part->derived_from_content_identity)))
            goto malformed;
        if (part->storage == YVEX_CONTENT_INLINE) {
            if (part->byte_count > reader.count - reader.offset ||
                part->byte_count > SIZE_MAX) goto malformed;
            part->bytes = malloc((size_t)part->byte_count);
            if (!part->bytes) goto no_memory;
            memcpy((void *)part->bytes, reader.bytes + reader.offset,
                   (size_t)part->byte_count);
            reader.offset += (size_t)part->byte_count;
        }
    }
    if (reader.offset != reader.count ||
        yvex_content_parts_validate(owned, item_count, err) != YVEX_OK)
        goto malformed;
    *parts = owned;
    *count = item_count;
    yvex_error_clear(err);
    return YVEX_OK;
malformed:
    yvex_content_parts_close(&owned, item_count);
    return content_refuse(err, YVEX_ERR_FORMAT,
                          "ordered content wire bytes are malformed");
no_memory:
    yvex_content_parts_close(&owned, item_count);
    return content_refuse(err, YVEX_ERR_NOMEM,
                          "content payload allocation failed");
}

int yvex_model_capability_validate(
    const yvex_model_capability_summary *capability, yvex_error *err)
{
    unsigned long long kinds = (1ull << YVEX_CONTENT_KIND_COUNT) - 1ull;
    unsigned long long properties =
        YVEX_MODEL_CAPABILITY_ORDERED_INPUT_PARTS |
        YVEX_MODEL_CAPABILITY_STATEFUL_SESSION |
        YVEX_MODEL_CAPABILITY_STREAMING_OUTPUT |
        YVEX_MODEL_CAPABILITY_DEMAND_ACTIVATION;
    if (!capability ||
        capability->schema_version != YVEX_MODEL_CAPABILITY_SCHEMA_V1 ||
        !capability->input_kinds || !capability->output_kinds ||
        (capability->input_kinds & ~kinds) ||
        (capability->output_kinds & ~kinds) ||
        (capability->execution_properties & ~properties) ||
        !capability->maximum_input_parts ||
        capability->maximum_input_parts > YVEX_CONTENT_MAX_PARTS)
        return content_refuse(err, YVEX_ERR_INVALID_ARG,
                              "complete model capability facts are required");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_capability_admit(
    const yvex_model_capability_summary *capability,
    const yvex_content_part *parts, unsigned long long count,
    yvex_error *err)
{
    unsigned long long index;
    if (yvex_model_capability_validate(capability, err) != YVEX_OK ||
        yvex_content_parts_validate(parts, count, err) != YVEX_OK)
        return yvex_error_code(err);
    if (count > capability->maximum_input_parts ||
        (count > 1u && !(capability->execution_properties &
                        YVEX_MODEL_CAPABILITY_ORDERED_INPUT_PARTS)))
        return content_refuse(err, YVEX_ERR_UNSUPPORTED,
                              "ordered content exceeds model capability");
    for (index = 0u; index < count; ++index)
        if (!(capability->input_kinds &
              YVEX_CONTENT_KIND_MASK(parts[index].kind)))
            return content_refuse(err, YVEX_ERR_UNSUPPORTED,
                                  "content kind is unsupported by this specialization");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_capability_profile_describe(
    yvex_model_capability_profile profile,
    yvex_model_capability_summary *capability, yvex_error *err)
{
    if (!capability ||
        (profile != YVEX_MODEL_CAPABILITY_PROFILE_TEXT_GENERATION &&
         profile !=
             YVEX_MODEL_CAPABILITY_PROFILE_CONDITIONED_AUDIOVISUAL_GENERATION))
        return content_refuse(err, YVEX_ERR_INVALID_ARG,
                              "known capability profile and output are required");
    memset(capability, 0, sizeof(*capability));
    capability->schema_version = YVEX_MODEL_CAPABILITY_SCHEMA_V1;
    capability->execution_properties = YVEX_MODEL_CAPABILITY_DEMAND_ACTIVATION;
    if (profile == YVEX_MODEL_CAPABILITY_PROFILE_TEXT_GENERATION) {
        capability->input_kinds = YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_TEXT);
        capability->output_kinds = YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_TEXT);
        capability->execution_properties |=
            YVEX_MODEL_CAPABILITY_ORDERED_INPUT_PARTS |
            YVEX_MODEL_CAPABILITY_STATEFUL_SESSION |
            YVEX_MODEL_CAPABILITY_STREAMING_OUTPUT;
        capability->maximum_input_parts = YVEX_CONTENT_MAX_PARTS;
    } else {
        capability->input_kinds = YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_TEXT) |
                                  YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_IMAGE);
        capability->output_kinds = YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_VIDEO) |
                                   YVEX_CONTENT_KIND_MASK(YVEX_CONTENT_AUDIO);
        capability->maximum_input_parts = 3u;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

const char *yvex_content_kind_name(yvex_content_kind kind)
{
    static const char *const names[YVEX_CONTENT_KIND_COUNT] = {
        "text", "image", "audio", "video", "file", "tensor"};
    return kind < YVEX_CONTENT_KIND_COUNT ? names[kind] : "unknown";
}

int yvex_model_capability_wire_encode(
    const yvex_model_capability_summary *capability,
    unsigned char output[40], yvex_error *err)
{
    content_writer writer = {output, 40u, 0u};
    if (!output || yvex_model_capability_validate(capability, err) != YVEX_OK)
        return yvex_error_code(err);
    if (!write_u64(&writer, capability->schema_version) ||
        !write_u64(&writer, capability->input_kinds) ||
        !write_u64(&writer, capability->output_kinds) ||
        !write_u64(&writer, capability->execution_properties) ||
        !write_u64(&writer, capability->maximum_input_parts))
        return content_refuse(err, YVEX_ERR_STATE,
                              "capability wire encoding failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_model_capability_wire_decode(
    const unsigned char input[40], unsigned long long byte_count,
    yvex_model_capability_summary *capability, yvex_error *err)
{
    content_reader reader = {input, (size_t)byte_count, 0u};
    unsigned long long schema;
    if (!input || byte_count != 40u || !capability) return content_refuse(
        err, YVEX_ERR_INVALID_ARG, "exact capability wire bytes are required");
    memset(capability, 0, sizeof(*capability));
    if (!read_u64(&reader, &schema) || schema > UINT_MAX ||
        !read_u64(&reader, &capability->input_kinds) ||
        !read_u64(&reader, &capability->output_kinds) ||
        !read_u64(&reader, &capability->execution_properties) ||
        !read_u64(&reader, &capability->maximum_input_parts))
        return content_refuse(err, YVEX_ERR_FORMAT,
                              "capability wire bytes are malformed");
    capability->schema_version = (unsigned int)schema;
    return yvex_model_capability_validate(capability, err);
}
