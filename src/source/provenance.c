/*
 * Bind pinned source identity to local provider evidence.
 *
 * Pinned revisions and index OIDs fail closed; payload is not hashed. Provider provenance is not
 * full shard payload trust.
 */
#define _XOPEN_SOURCE 700
#include <errno.h>
#include <fcntl.h>
#include <ctype.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

static int provenance_refuse(yvex_error *err,
                             yvex_status status,
                             const char *where,
                             const char *message) {
    yvex_error_set(err, status, where, message);
    return status;
}

#define SOURCE_MANIFEST_CAP (32u * 1024u * 1024u)
#define SOURCE_ACQUISITION_MANIFEST_CAP (4u * 1024u * 1024u)

struct yvex_source_acquisition {
    yvex_source_acquisition_facts facts;
    yvex_source_acquisition_file *files;
};

static const yvex_source_target_identity release_source_identity = {
    YVEX_SOURCE_RELEASE_TARGET_ID,
    YVEX_SOURCE_RELEASE_FAMILY_KEY,
    YVEX_SOURCE_RELEASE_FAMILY_DISPLAY,
    YVEX_SOURCE_RELEASE_NAME,
    YVEX_SOURCE_RELEASE_REPOSITORY,
    YVEX_SOURCE_RELEASE_SOURCE_LEAF,
    YVEX_SOURCE_RELEASE_REVISION,
    YVEX_SOURCE_RELEASE_INDEX_PATH,
    YVEX_SOURCE_RELEASE_INDEX_OID,
    YVEX_SOURCE_RELEASE_INDEX_SIZE,
    YVEX_SOURCE_RELEASE_INVENTORY_AUTHORITY,
    YVEX_SOURCE_RELEASE_CONFIG_TYPE,
    YVEX_SOURCE_RELEASE_CONFIG_ARCHITECTURE,
};

/*
 * Expose the immutable release source identity owned by provenance.
 *
 * Releases only resources owned by source provenance; cleanup remains deterministic.
 */
const yvex_source_target_identity *yvex_source_release_identity(void) {
    return &release_source_identity;
}

/*
 * Test target equality without importing model-catalog policy.
 *
 * Releases only resources owned by source provenance; cleanup remains deterministic.
 */
int yvex_source_is_release_target(const char *target_id) {
    return target_id && strcmp(target_id, release_source_identity.target_id) == 0;
}

/* Derive the canonical source directory for an admitted identity. */
int yvex_source_target_path(char *out,
                            size_t cap,
                            const char *models_root,
                            const yvex_source_target_identity *identity) {
    int n;

    if (!out || cap == 0u || !models_root || !models_root[0] || !identity ||
        !identity->family_key || !identity->source_dir_leaf) {
        return 0;
    }
    n = snprintf(
        out, cap, "%s/hf/%s/%s", models_root, identity->family_key, identity->source_dir_leaf);
    return n >= 0 && (size_t)n < cap;
}

typedef struct {
    uint32_t state[5];
    unsigned long long length;
    unsigned char block[64];
    size_t block_length;
} source_sha1;

static uint32_t source_sha1_rotl(uint32_t value, unsigned int bits) {
    return (value << bits) | (value >> (32u - bits));
}

/* Applies one SHA-1 compression block to caller-owned Git identity state. */
static void source_sha1_transform(source_sha1 *ctx, const unsigned char block[64]) {
    uint32_t words[80];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    unsigned int i;

    for (i = 0u; i < 16u; ++i) {
        words[i] = ((uint32_t)block[i * 4u] << 24u) | ((uint32_t)block[i * 4u + 1u] << 16u) |
                   ((uint32_t)block[i * 4u + 2u] << 8u) | (uint32_t)block[i * 4u + 3u];
    }
    for (i = 16u; i < 80u; ++i) {
        words[i] =
            source_sha1_rotl(words[i - 3u] ^ words[i - 8u] ^ words[i - 14u] ^ words[i - 16u], 1u);
    }
    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    for (i = 0u; i < 80u; ++i) {
        uint32_t f;
        uint32_t k;
        uint32_t temp;

        if (i < 20u) {
            f = (b & c) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40u) {
            f = b ^ c ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60u) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ c ^ d;
            k = 0xca62c1d6u;
        }
        temp = source_sha1_rotl(a, 5u) + f + e + k + words[i];
        e = d;
        d = c;
        c = source_sha1_rotl(b, 30u);
        b = a;
        a = temp;
    }
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
}

static void source_sha1_init(source_sha1 *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->state[0] = 0x67452301u;
    ctx->state[1] = 0xefcdab89u;
    ctx->state[2] = 0x98badcfeu;
    ctx->state[3] = 0x10325476u;
    ctx->state[4] = 0xc3d2e1f0u;
}

static void source_sha1_update(source_sha1 *ctx, const unsigned char *data, size_t length) {
    size_t offset = 0u;

    ctx->length += (unsigned long long)length;
    while (offset < length) {
        size_t available = sizeof(ctx->block) - ctx->block_length;
        size_t take = length - offset < available ? length - offset : available;
        memcpy(ctx->block + ctx->block_length, data + offset, take);
        ctx->block_length += take;
        offset += take;
        if (ctx->block_length == sizeof(ctx->block)) {
            source_sha1_transform(ctx, ctx->block);
            ctx->block_length = 0u;
        }
    }
}

static void source_sha1_final(source_sha1 *ctx, unsigned char digest[20]) {
    unsigned long long bits = ctx->length * 8ull;
    unsigned int i;

    ctx->block[ctx->block_length++] = 0x80u;
    if (ctx->block_length > 56u) {
        memset(ctx->block + ctx->block_length, 0, sizeof(ctx->block) - ctx->block_length);
        source_sha1_transform(ctx, ctx->block);
        ctx->block_length = 0u;
    }
    memset(ctx->block + ctx->block_length, 0, 56u - ctx->block_length);
    for (i = 0u; i < 8u; ++i) {
        ctx->block[63u - i] = (unsigned char)((bits >> (i * 8u)) & 0xffu);
    }
    source_sha1_transform(ctx, ctx->block);
    for (i = 0u; i < 5u; ++i) {
        digest[i * 4u] = (unsigned char)(ctx->state[i] >> 24u);
        digest[i * 4u + 1u] = (unsigned char)(ctx->state[i] >> 16u);
        digest[i * 4u + 2u] = (unsigned char)(ctx->state[i] >> 8u);
        digest[i * 4u + 3u] = (unsigned char)ctx->state[i];
    }
}

/* Computes Git's SHA-1 blob identity for one bounded metadata file. */
int yvex_source_git_blob_oid_file(const char *path, char out_hex[41], yvex_error *err) {
    unsigned long long size;
    char header[64];
    int header_length;
    FILE *fp;
    unsigned char buffer[16384];
    unsigned char digest[20];
    source_sha1 ctx;
    size_t got;
    unsigned int i;
    int read_failed;
    static const char hex[] = "0123456789abcdef";

    if (!path || !out_hex || !yvex_source_regular_file(path, &size)) {
        return provenance_refuse(err, YVEX_ERR_INVALID_ARG, "source_git_blob_oid",
            "a regular metadata file and output are required");
    }
    header_length = snprintf(header, sizeof(header), "blob %llu", size);
    if (header_length < 0 || (size_t)header_length + 1u > sizeof(header)) {
        return provenance_refuse(err, YVEX_ERR_BOUNDS, "source_git_blob_oid", "Git blob header overflow");
    }
    fp = fopen(path, "rb");
    if (!fp) {
        yvex_error_setf(
            err, YVEX_ERR_IO, "source_git_blob_oid", "cannot open metadata file: %s", path);
        return YVEX_ERR_IO;
    }
    source_sha1_init(&ctx);
    source_sha1_update(&ctx, (const unsigned char *)header, (size_t)header_length + 1u);
    while ((got = fread(buffer, 1u, sizeof(buffer), fp)) > 0u) {
        source_sha1_update(&ctx, buffer, got);
    }
    read_failed = ferror(fp);
    if (fclose(fp) != 0)
        read_failed = 1;
    if (read_failed) {
        yvex_error_setf(
            err, YVEX_ERR_IO, "source_git_blob_oid", "cannot read metadata file: %s", path);
        return YVEX_ERR_IO;
    }
    source_sha1_final(&ctx, digest);
    for (i = 0u; i < 20u; ++i) {
        out_hex[i * 2u] = hex[digest[i] >> 4u];
        out_hex[i * 2u + 1u] = hex[digest[i] & 0x0fu];
    }
    out_hex[40] = '\0';
    return YVEX_OK;
}

/* Computes Git's blob identity over the exact bytes retained by a caller. */
static int
source_git_blob_oid_bytes(const unsigned char *bytes, size_t byte_count, char out_hex[41]) {
    char header[64];
    int header_length;
    unsigned char digest[20];
    source_sha1 ctx;
    unsigned int index;
    static const char hex[] = "0123456789abcdef";

    if ((!bytes && byte_count) || !out_hex)
        return 0;
    header_length = snprintf(header, sizeof(header), "blob %llu", (unsigned long long)byte_count);
    if (header_length < 0 || (size_t)header_length + 1u > sizeof(header))
        return 0;
    source_sha1_init(&ctx);
    source_sha1_update(&ctx, (const unsigned char *)header, (size_t)header_length + 1u);
    source_sha1_update(&ctx, bytes, byte_count);
    source_sha1_final(&ctx, digest);
    for (index = 0u; index < sizeof(digest); ++index) {
        out_hex[index * 2u] = hex[digest[index] >> 4u];
        out_hex[index * 2u + 1u] = hex[digest[index] & 0x0fu];
    }
    out_hex[40] = '\0';
    return 1;
}

typedef enum {
    MANIFEST_TEXT,
    MANIFEST_U64,
    MANIFEST_NULLABLE_TEXT,
    MANIFEST_PAYLOAD_SHARDS
} manifest_field_kind;

typedef struct {
    const char *key;
    unsigned int bit;
    manifest_field_kind kind;
    size_t offset;
    size_t size;
} manifest_field;

typedef struct {
    unsigned long long id;
    unsigned long long file_bytes;
    unsigned long long data_region_offset;
    unsigned long long payload_bytes;
    char name[YVEX_PATH_CAP];
    char algorithm[24];
    char authority[40];
    char expected[65];
    char observed[65];
    char trust[40];
} manifest_payload_shard;

typedef struct {
    char identity[65];
    char trust_class[40];
    char digest_algorithm[24];
    unsigned long long source_identity;
    unsigned long long shard_count;
    unsigned long long shard_file_bytes;
    unsigned long long tensor_count;
    unsigned long long logical_tensor_bytes;
    unsigned long long parsed_shards;
    unsigned long long parsed_file_bytes;
    int parsed_trust_class;
} manifest_payload;

static int source_manifest_parse_payload_shards(yvex_json *json,
                                                unsigned long long *count,
                                                unsigned long long *total_file_bytes);

static const manifest_field manifest_source_fields[] = {
    {"kind", 1u, MANIFEST_TEXT, offsetof(yvex_source_verification, source_kind),
     sizeof(((yvex_source_verification *)0)->source_kind)},
    {"repo", 2u, MANIFEST_TEXT, offsetof(yvex_source_verification, repository_id),
     sizeof(((yvex_source_verification *)0)->repository_id)},
    {"revision", 4u, MANIFEST_TEXT, offsetof(yvex_source_verification, manifest_revision),
     sizeof(((yvex_source_verification *)0)->manifest_revision)},
};

static const manifest_field manifest_target_fields[] = {
    {"id", 1u, MANIFEST_TEXT, offsetof(yvex_source_verification, manifest_target_id),
     sizeof(((yvex_source_verification *)0)->manifest_target_id)},
};

static const manifest_field manifest_verification_fields[] = {
    {"stage", 1u, MANIFEST_TEXT, offsetof(yvex_source_verification, verification_stage),
     sizeof(((yvex_source_verification *)0)->verification_stage)},
    {"inventory_authority", 2u, MANIFEST_TEXT,
     offsetof(yvex_source_verification, inventory_authority),
     sizeof(((yvex_source_verification *)0)->inventory_authority)},
    {"source_file_count", 4u, MANIFEST_U64,
     offsetof(yvex_source_verification, manifest_source_file_count), 0u},
    {"source_total_bytes", 8u, MANIFEST_U64,
     offsetof(yvex_source_verification, manifest_source_total_bytes), 0u},
    {"shard_count", 16u, MANIFEST_U64,
     offsetof(yvex_source_verification, manifest_shard_count), 0u},
    {"shard_bytes", 32u, MANIFEST_U64,
     offsetof(yvex_source_verification, manifest_shard_bytes), 0u},
    {"header_tensor_count", 64u, MANIFEST_U64,
     offsetof(yvex_source_verification, manifest_header_tensor_count), 0u},
    {"config_status", 128u, MANIFEST_TEXT,
     offsetof(yvex_source_verification, manifest_config_status),
     sizeof(((yvex_source_verification *)0)->manifest_config_status)},
    {"tokenizer_status", 256u, MANIFEST_TEXT,
     offsetof(yvex_source_verification, manifest_tokenizer_status),
     sizeof(((yvex_source_verification *)0)->manifest_tokenizer_status)},
    {"payload_digest_status", 512u, MANIFEST_TEXT,
     offsetof(yvex_source_verification, manifest_payload_digest_status),
     sizeof(((yvex_source_verification *)0)->manifest_payload_digest_status)},
    {"upstream_index_oid", 1024u, MANIFEST_TEXT,
     offsetof(yvex_source_verification, upstream_index_oid),
     sizeof(((yvex_source_verification *)0)->upstream_index_oid)},
};

static const manifest_field manifest_payload_shard_fields[] = {
    {"id", 1u, MANIFEST_U64, offsetof(manifest_payload_shard, id), 0u},
    {"name", 2u, MANIFEST_TEXT, offsetof(manifest_payload_shard, name),
     sizeof(((manifest_payload_shard *)0)->name)},
    {"file_bytes", 4u, MANIFEST_U64, offsetof(manifest_payload_shard, file_bytes), 0u},
    {"data_region_offset", 8u, MANIFEST_U64,
     offsetof(manifest_payload_shard, data_region_offset), 0u},
    {"payload_bytes", 16u, MANIFEST_U64, offsetof(manifest_payload_shard, payload_bytes), 0u},
    {"digest_algorithm", 32u, MANIFEST_TEXT, offsetof(manifest_payload_shard, algorithm),
     sizeof(((manifest_payload_shard *)0)->algorithm)},
    {"digest_authority", 64u, MANIFEST_TEXT, offsetof(manifest_payload_shard, authority),
     sizeof(((manifest_payload_shard *)0)->authority)},
    {"expected_digest", 128u, MANIFEST_NULLABLE_TEXT,
     offsetof(manifest_payload_shard, expected), sizeof(((manifest_payload_shard *)0)->expected)},
    {"observed_digest", 256u, MANIFEST_TEXT, offsetof(manifest_payload_shard, observed),
     sizeof(((manifest_payload_shard *)0)->observed)},
    {"trust_class", 512u, MANIFEST_TEXT, offsetof(manifest_payload_shard, trust),
     sizeof(((manifest_payload_shard *)0)->trust)},
};

static const manifest_field manifest_payload_fields[] = {
    {"identity", 1u, MANIFEST_TEXT, offsetof(manifest_payload, identity),
     sizeof(((manifest_payload *)0)->identity)},
    {"trust_class", 2u, MANIFEST_TEXT, offsetof(manifest_payload, trust_class),
     sizeof(((manifest_payload *)0)->trust_class)},
    {"digest_algorithm", 4u, MANIFEST_TEXT, offsetof(manifest_payload, digest_algorithm),
     sizeof(((manifest_payload *)0)->digest_algorithm)},
    {"source_snapshot_identity", 8u, MANIFEST_U64,
     offsetof(manifest_payload, source_identity), 0u},
    {"shard_count", 16u, MANIFEST_U64, offsetof(manifest_payload, shard_count), 0u},
    {"shard_file_bytes", 32u, MANIFEST_U64,
     offsetof(manifest_payload, shard_file_bytes), 0u},
    {"tensor_count", 64u, MANIFEST_U64, offsetof(manifest_payload, tensor_count), 0u},
    {"logical_tensor_bytes", 128u, MANIFEST_U64,
     offsetof(manifest_payload, logical_tensor_bytes), 0u},
    {"shards", 256u, MANIFEST_PAYLOAD_SHARDS, 0u, 0u},
};

static int manifest_field_parse(yvex_json *json, const manifest_field *field, void *object) {
    unsigned char *base = (unsigned char *)object;

    if (field->kind == MANIFEST_TEXT)
        return yvex_json_string(json, (char *)(base + field->offset), field->size);
    if (field->kind == MANIFEST_U64)
        return yvex_json_u64(json, (unsigned long long *)(void *)(base + field->offset));
    if (field->kind == MANIFEST_PAYLOAD_SHARDS) {
        manifest_payload *payload = (manifest_payload *)object;

        payload->parsed_trust_class = source_manifest_parse_payload_shards(
            json, &payload->parsed_shards, &payload->parsed_file_bytes);
        return payload->parsed_trust_class != 0;
    }
    yvex_json_space(json);
    if (json->cursor < json->end && *json->cursor == '"')
        return yvex_json_string(json, (char *)(base + field->offset), field->size);
    if ((size_t)(json->end - json->cursor) < 4u || memcmp(json->cursor, "null", 4u) != 0)
        return 0;
    json->cursor += 4u;
    return 1;
}

static int manifest_object_parse(yvex_json *json,
                                 const manifest_field *fields,
                                 size_t field_count,
                                 unsigned int required,
                                 void *object) {
    char key[YVEX_JSON_KEY_CAP];
    unsigned int seen = 0u;
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        const manifest_field *field;

        field = yvex_core_keyed_row_find(
            fields, field_count, sizeof(fields[0]), offsetof(manifest_field, key), key);
        if (!field) {
            if (!yvex_json_skip_value(json))
                return 0;
        } else if ((seen & field->bit) || !manifest_field_parse(json, field, object)) {
            return 0;
        } else {
            seen |= field->bit;
        }
    }
    return item == YVEX_JSON_ITEM_END && (seen & required) == required;
}

static int source_manifest_parse_source(yvex_json *json, yvex_source_verification *out) {
    return manifest_object_parse(json,
                                 manifest_source_fields,
                                 sizeof(manifest_source_fields) / sizeof(manifest_source_fields[0]),
                                 2u,
                                 out);
}

static int source_manifest_parse_local(yvex_json *json, char *path, size_t cap) {
    char key[YVEX_JSON_KEY_CAP];
    int seen = 0;
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "path") == 0) {
            if (seen || !yvex_json_string(json, path, cap))
                return 0;
            seen = 1;
        } else if (!yvex_json_skip_value(json)) {
            return 0;
        }
    }
    return item == YVEX_JSON_ITEM_END && seen;
}

/* Parses the canonical target identity declared by a verifier-owned manifest. */

static int source_manifest_parse_target(yvex_json *json, yvex_source_verification *out) {
    return manifest_object_parse(json,
                                 manifest_target_fields,
                                 sizeof(manifest_target_fields) / sizeof(manifest_target_fields[0]),
                                 1u,
                                 out);
}

static int source_manifest_parse_verification(yvex_json *json, yvex_source_verification *out) {
    return manifest_object_parse(
        json,
        manifest_verification_fields,
        sizeof(manifest_verification_fields) / sizeof(manifest_verification_fields[0]),
        2047u,
        out);
}

static int source_manifest_parse_payload_shard(yvex_json *json,
                                               unsigned long long expected_id,
                                               char previous_name[YVEX_PATH_CAP],
                                               unsigned long long *file_bytes_out,
                                               int *upstream_trust_out) {
    manifest_payload_shard shard;

    memset(&shard, 0, sizeof(shard));
    shard.id = ULLONG_MAX;
    if (!manifest_object_parse(json,
                               manifest_payload_shard_fields,
                               sizeof(manifest_payload_shard_fields) /
                                   sizeof(manifest_payload_shard_fields[0]),
                               1023u,
                               &shard) ||
        shard.id != expected_id || !yvex_source_payload_name_is_canonical(shard.name) ||
        (previous_name[0] && strcmp(previous_name, shard.name) >= 0) ||
        shard.file_bytes == 0u || shard.data_region_offset > shard.file_bytes ||
        shard.payload_bytes != shard.file_bytes - shard.data_region_offset ||
        strcmp(shard.algorithm, "sha256") != 0 || !shard.authority[0] ||
        !yvex_sha256_hex_valid(shard.observed) ||
        (shard.expected[0] &&
         (!yvex_sha256_hex_valid(shard.expected) ||
          strcmp(shard.expected, shard.observed) != 0)) ||
        ((shard.expected[0] && strcmp(shard.trust, "upstream_payload_verified") != 0) ||
         (!shard.expected[0] &&
          (strcmp(shard.trust, "local_payload_snapshot_sealed") != 0 ||
           strcmp(shard.authority, "local-snapshot-seal") != 0)))) {
        return 0;
    }
    if (snprintf(previous_name, YVEX_PATH_CAP, "%s", shard.name) < 0)
        return 0;
    *file_bytes_out = shard.file_bytes;
    *upstream_trust_out = shard.expected[0] != '\0';
    return 1;
}

/* Parses deterministic canonical-order shard rows and returns their exact count. */
static int source_manifest_parse_payload_shards(yvex_json *json,
                                                unsigned long long *count,
                                                unsigned long long *total_file_bytes) {
    unsigned long long index = 0u;
    unsigned long long total = 0u;
    char previous_name[YVEX_PATH_CAP] = "";
    int all_upstream_trust = 1;
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY))
        return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        unsigned long long file_bytes;
        int upstream_trust;

        if (!source_manifest_parse_payload_shard(
                json, index, previous_name, &file_bytes, &upstream_trust) ||
            !yvex_core_u64_add(total, file_bytes, &total))
            return 0;
        if (!upstream_trust)
            all_upstream_trust = 0;
        index++;
    }
    if (item != YVEX_JSON_ITEM_END || iter.trailing_separator || index == 0u)
        return 0;
    *count = index;
    *total_file_bytes = total;
    return all_upstream_trust ? 2 : 1;
}

/* Parses v3 aggregate payload identity and validates all published shard rows. */

static int source_manifest_parse_payload(yvex_json *json, yvex_source_verification *out) {
    manifest_payload payload;

    memset(&payload, 0, sizeof(payload));
    if (!manifest_object_parse(json,
                               manifest_payload_fields,
                               sizeof(manifest_payload_fields) /
                                   sizeof(manifest_payload_fields[0]),
                               511u,
                               &payload) ||
        !yvex_sha256_hex_valid(payload.identity) ||
        strcmp(payload.digest_algorithm, "sha256") != 0 ||
        (strcmp(payload.trust_class, "upstream_payload_verified") != 0 &&
         strcmp(payload.trust_class, "local_payload_snapshot_sealed") != 0) ||
        payload.source_identity == 0u || payload.shard_count != payload.parsed_shards ||
        payload.shard_file_bytes != payload.parsed_file_bytes ||
        ((payload.parsed_trust_class == 2 &&
          strcmp(payload.trust_class, "upstream_payload_verified") != 0) ||
         (payload.parsed_trust_class == 1 &&
          strcmp(payload.trust_class, "local_payload_snapshot_sealed") != 0))) {
        return 0;
    }
    snprintf(out->manifest_payload_identity,
             sizeof(out->manifest_payload_identity),
             "%s",
             payload.identity);
    snprintf(out->manifest_payload_trust_class,
             sizeof(out->manifest_payload_trust_class),
             "%s",
             payload.trust_class);
    snprintf(out->manifest_payload_digest_algorithm,
             sizeof(out->manifest_payload_digest_algorithm),
             "%s",
             payload.digest_algorithm);
    out->manifest_payload_source_snapshot_identity = payload.source_identity;
    out->manifest_payload_shard_count = payload.shard_count;
    out->manifest_payload_bytes = payload.shard_file_bytes;
    out->manifest_payload_tensor_count = payload.tensor_count;
    out->manifest_payload_logical_tensor_bytes = payload.logical_tensor_bytes;
    return 1;
}

static int source_manifest_parse(const char *data,
                                 size_t length,
                                 yvex_source_verification *out,
                                 char *local_path,
                                 size_t local_path_cap) {
    yvex_json json;
    yvex_json_iter iter;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];
    unsigned int seen = 0u;

    yvex_json_init(&json, data, length);
    if (!yvex_json_iter_begin(&json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "schema") == 0) {
            if ((seen & 1u) ||
                !yvex_json_string(&json, out->manifest_schema, sizeof(out->manifest_schema)))
                return 0;
            seen |= 1u;
        } else if (strcmp(key, "status") == 0) {
            if ((seen & 2u) ||
                !yvex_json_string(&json, out->manifest_status, sizeof(out->manifest_status)))
                return 0;
            seen |= 2u;
        } else if (strcmp(key, "source") == 0) {
            if ((seen & 4u) || !source_manifest_parse_source(&json, out))
                return 0;
            seen |= 4u;
        } else if (strcmp(key, "local") == 0) {
            if ((seen & 8u) || !source_manifest_parse_local(&json, local_path, local_path_cap))
                return 0;
            seen |= 8u;
        } else if (strcmp(key, "target") == 0) {
            if ((seen & 16u) || !source_manifest_parse_target(&json, out))
                return 0;
            seen |= 16u;
        } else if (strcmp(key, "verification") == 0) {
            if ((seen & 32u) || !source_manifest_parse_verification(&json, out))
                return 0;
            seen |= 32u;
        } else if (strcmp(key, "payload") == 0) {
            if ((seen & 64u) || !source_manifest_parse_payload(&json, out))
                return 0;
            seen |= 64u;
        } else if (!yvex_json_skip_value(&json)) {
            return 0;
        }
    }
    if (item != YVEX_JSON_ITEM_END || !yvex_json_complete(&json) || (seen & 15u) != 15u)
        return 0;
    if (strcmp(out->manifest_schema, "yvex.source_manifest.v1") == 0) {
        return seen == 15u;
    }
    if (strcmp(out->manifest_schema, "yvex.source_manifest.v2") == 0)
        return seen == 63u;
    if (strcmp(out->manifest_schema, "yvex.source_manifest.v3") == 0)
        return seen == 127u;
    return -1;
}

static int source_manifest_path(const yvex_source_verify_options *options, char *out, size_t cap) {
    char candidate[YVEX_PATH_CAP];
    int n;

    if (options->manifest_path && options->manifest_path[0]) {
        n = snprintf(out, cap, "%s", options->manifest_path);
        return n >= 0 && (size_t)n < cap;
    }
    if (options->promote_manifest && options->models_root) {
        n = snprintf(out,
                     cap,
                     "%s/gguf/%s/%s",
                     options->models_root,
                     options->identity->family_key,
                     YVEX_SOURCE_RELEASE_MANIFEST_LEAF);
        return n >= 0 && (size_t)n < cap;
    }
    if (yvex_source_path_join(
            candidate, sizeof(candidate), options->source_path, "source-manifest.json") &&
        yvex_source_regular_file(candidate, NULL)) {
        n = snprintf(out, cap, "%s", candidate);
        return n >= 0 && (size_t)n < cap;
    }
    if (yvex_source_path_join(
            candidate, sizeof(candidate), options->source_path, "source_manifest.json") &&
        yvex_source_regular_file(candidate, NULL)) {
        n = snprintf(out, cap, "%s", candidate);
        return n >= 0 && (size_t)n < cap;
    }
    n = options->models_root ? snprintf(out,
                                        cap,
                                        "%s/gguf/%s/%s",
                                        options->models_root,
                                        options->identity->family_key,
                                        YVEX_SOURCE_RELEASE_MANIFEST_LEAF)
                             : -1;
    return n >= 0 && (size_t)n < cap;
}

int yvex_source_provenance_manifest_read(const yvex_source_verify_options *options,
                                         yvex_source_verification *out,
                                         yvex_error *err) {
    char *data;
    size_t length;
    char manifest_local[YVEX_PATH_CAP] = "";
    char resolved_manifest_local[YVEX_PATH_CAP];

    if (!options || !out ||
        !source_manifest_path(options, out->manifest_path, sizeof(out->manifest_path)) ||
        !yvex_source_regular_file(out->manifest_path, NULL)) {
        yvex_source_verification_add_blocker(out, "missing-source-manifest");
        return YVEX_OK;
    }
    data = yvex_read_bounded_file(out->manifest_path, SOURCE_MANIFEST_CAP, &length, err);
    if (!data) {
        if (yvex_error_code(err) == YVEX_ERR_NOMEM)
            return YVEX_ERR_NOMEM;
        yvex_source_verification_add_blocker(out, "malformed-source-manifest");
        yvex_error_clear(err);
        return YVEX_OK;
    }
    {
        int parse_status =
            source_manifest_parse(data, length, out, manifest_local, sizeof(manifest_local));
        if (parse_status <= 0) {
            free(data);
            yvex_source_verification_add_blocker(out,
                                                 parse_status < 0
                                                     ? "unsupported-source-manifest-version"
                                                     : "malformed-source-manifest");
            return YVEX_OK;
        }
    }
    free(data);
    if (strcmp(out->source_kind, "huggingface") != 0) {
        yvex_source_verification_add_blocker(
            out, out->source_kind[0] ? "unsupported-source-kind" : "missing-source-kind");
    }
    if (strcmp(out->repository_id, options->identity->upstream_repo_id) != 0) {
        yvex_source_verification_add_blocker(out, "wrong-source-repository");
    } else {
        out->repository_verified = 1;
    }
    if (!realpath(manifest_local, resolved_manifest_local) ||
        strcmp(resolved_manifest_local, out->resolved_source_path) != 0) {
        yvex_source_verification_add_blocker(out, "wrong-source-local-path");
    }
    if (out->manifest_target_id[0] &&
        strcmp(out->manifest_target_id, options->identity->target_id) != 0) {
        yvex_source_verification_add_blocker(out, "wrong-source-target");
    }
    return YVEX_OK;
}

static int source_metadata_read(const char *source_path,
                                const char *name,
                                char *revision,
                                size_t revision_cap,
                                char *oid,
                                size_t oid_cap) {
    char cache[YVEX_PATH_CAP];
    char path[YVEX_PATH_CAP];
    FILE *fp;
    char line[160];
    int n;

    if (!yvex_source_path_join(cache, sizeof(cache), source_path, ".cache/huggingface/download"))
        return 0;
    n = snprintf(path, sizeof(path), "%s/%s.metadata", cache, name);
    if (n < 0 || (size_t)n >= sizeof(path))
        return 0;
    fp = fopen(path, "rb");
    if (!fp || !fgets(line, sizeof(line), fp)) {
        if (fp)
            fclose(fp);
        return 0;
    }
    line[strcspn(line, "\r\n")] = '\0';
    if (!yvex_source_revision_is_commit(line) || snprintf(revision, revision_cap, "%s", line) < 0) {
        fclose(fp);
        return 0;
    }
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }
    line[strcspn(line, "\r\n")] = '\0';
    if (!line[0] || snprintf(oid, oid_cap, "%s", line) < 0) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

int yvex_source_provenance_payload_digest(const yvex_source_verification *verification,
                                          const char *canonical_name,
                                          yvex_source_payload_digest_fact *out,
                                          yvex_error *err) {
    char revision[128];
    char oid[128];
    size_t index;

    if (!verification || !canonical_name || !out || !verification->resolved_source_path[0] ||
        !verification->revision[0]) {
        return provenance_refuse(err, YVEX_ERR_INVALID_ARG, "source_payload_digest_provenance",
            "verified source, canonical shard name, and output are required");
    }
    memset(out, 0, sizeof(*out));
    if (!source_metadata_read(verification->resolved_source_path,
                              canonical_name,
                              revision,
                              sizeof(revision),
                              oid,
                              sizeof(oid))) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    out->revision_matches = strcmp(revision, verification->revision) == 0;
    if (!out->revision_matches) {
        return provenance_refuse(err, YVEX_ERR_FORMAT, "source_payload_digest_provenance",
            "payload provider metadata revision does not match verified source");
    }
    if (strlen(oid) != 64u) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (!yvex_sha256_hex_valid(oid)) {
        return provenance_refuse(err, YVEX_ERR_FORMAT, "source_payload_digest_provenance",
            "64-byte provider digest is not valid SHA-256 hexadecimal");
    }
    for (index = 0u; index < 64u; ++index)
        out->expected_digest[index] = (char)tolower((unsigned char)oid[index]);
    out->expected_digest[64] = '\0';
    snprintf(out->algorithm, sizeof(out->algorithm), "%s", "sha256");
    snprintf(out->authority, sizeof(out->authority), "%s", "huggingface-git-lfs-etag");
    out->available = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int source_oid_is_sha1(const char *oid) {
    size_t i;

    if (!oid || strlen(oid) != 40u)
        return 0;
    for (i = 0u; i < 40u; ++i) {
        if (!isxdigit((unsigned char)oid[i]))
            return 0;
    }
    return 1;
}

static int source_metadata_name_valid(const char *name) {
    const char *cursor;

    if (!name || !name[0] || name[0] == '/' || strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
        return 0;
    for (cursor = name; *cursor; ++cursor)
        if (*cursor == '/' || *cursor == '\\' || *cursor == '\n' || *cursor == '\r')
            return 0;
    return 1;
}

/* Bind one bounded sidecar to its pinned revision and provider Git blob identity. */
static int source_metadata_identity(const yvex_source_verification *verification,
                                    const char *canonical_name,
                                    yvex_source_metadata_identity_fact *out,
                                    yvex_error *err) {
    yvex_source_metadata_identity_fact fact;
    char provider_revision[128];
    char provider_oid[128];
    char path[YVEX_PATH_CAP];
    unsigned long long file_bytes;
    int rc;

    if (out)
        memset(out, 0, sizeof(*out));
    if (!verification || !out || !verification->resolved_source_path[0] ||
        !verification->revision[0] || !source_metadata_name_valid(canonical_name)) {
        return provenance_refuse(err, YVEX_ERR_INVALID_ARG, "source_metadata_identity",
            "verified source and canonical metadata name are required");
    }
    memset(&fact, 0, sizeof(fact));
    if (!source_metadata_read(verification->resolved_source_path,
                              canonical_name,
                              provider_revision,
                              sizeof(provider_revision),
                              provider_oid,
                              sizeof(provider_oid)) ||
        !source_oid_is_sha1(provider_oid)) {
        return provenance_refuse(err, YVEX_ERR_FORMAT, "source_metadata_identity",
            "provider metadata lacks a pinned Git blob identity");
    }
    fact.revision_matches = strcmp(provider_revision, verification->revision) == 0;
    if (!fact.revision_matches) {
        return provenance_refuse(err, YVEX_ERR_FORMAT, "source_metadata_identity",
            "metadata provider revision differs from source snapshot");
    }
    if (!yvex_source_path_join(
            path, sizeof(path), verification->resolved_source_path, canonical_name) ||
        !yvex_source_regular_file(path, &file_bytes)) {
        return provenance_refuse(err, YVEX_ERR_IO, "source_metadata_identity",
            "metadata sidecar is missing or not a regular file");
    }
    rc = yvex_source_git_blob_oid_file(path, fact.observed_git_blob_oid, err);
    if (rc != YVEX_OK)
        return rc;
    if (strcmp(provider_oid, fact.observed_git_blob_oid) != 0) {
        return provenance_refuse(err, YVEX_ERR_FORMAT, "source_metadata_identity",
            "metadata sidecar Git blob identity mismatch");
    }
    yvex_core_text_copy(fact.canonical_name, sizeof(fact.canonical_name), canonical_name);
    memcpy(fact.revision, provider_revision, 41u);
    memcpy(fact.expected_git_blob_oid, provider_oid, 41u);
    fact.file_bytes = file_bytes;
    fact.identity_verified = 1;
    *out = fact;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Retain one identity-verified sidecar in bounded memory after a post-read check. */
int yvex_source_provenance_metadata_read(const yvex_source_verification *verification,
                                         const char *canonical_name,
                                         size_t maximum_bytes,
                                         yvex_source_metadata_blob *out,
                                         yvex_error *err) {
    yvex_source_metadata_blob blob;
    yvex_source_metadata_identity_fact after;
    char path[YVEX_PATH_CAP];
    char retained_oid[41];
    FILE *fp;
    size_t got;
    int read_failed;
    int rc;

    if (out)
        memset(out, 0, sizeof(*out));
    if (!verification || !out || !maximum_bytes) {
        return provenance_refuse(err, YVEX_ERR_INVALID_ARG, "source_metadata_read",
            "verified source, byte budget, and output are required");
    }
    memset(&blob, 0, sizeof(blob));
    rc = source_metadata_identity(verification, canonical_name, &blob.identity, err);
    if (rc != YVEX_OK)
        return rc;
    if (blob.identity.file_bytes > maximum_bytes || blob.identity.file_bytes > SIZE_MAX) {
        return provenance_refuse(err, YVEX_ERR_BOUNDS, "source_metadata_read",
            "metadata sidecar exceeds its configured byte budget");
    }
    blob.byte_count = (size_t)blob.identity.file_bytes;
    blob.bytes = (unsigned char *)malloc(blob.byte_count ? blob.byte_count : 1u);
    if (!blob.bytes) {
        return provenance_refuse(err, YVEX_ERR_NOMEM, "source_metadata_read",
            "metadata sidecar buffer allocation failed");
    }
    if (!yvex_source_path_join(
            path, sizeof(path), verification->resolved_source_path, canonical_name)) {
        yvex_source_metadata_blob_release(&blob);
        return provenance_refuse(err, YVEX_ERR_BOUNDS, "source_metadata_read",
            "metadata sidecar path construction overflowed");
    }
    fp = fopen(path, "rb");
    if (!fp) {
        yvex_source_metadata_blob_release(&blob);
        return provenance_refuse(err, YVEX_ERR_IO, "source_metadata_read", "metadata sidecar open failed");
    }
    got = blob.byte_count ? fread(blob.bytes, 1u, blob.byte_count, fp) : 0u;
    read_failed = got != blob.byte_count || ferror(fp);
    if (fclose(fp) != 0)
        read_failed = 1;
    if (read_failed) {
        yvex_source_metadata_blob_release(&blob);
        yvex_error_set(
            err, YVEX_ERR_IO, "source_metadata_read", "metadata sidecar exact read failed");
        return YVEX_ERR_IO;
    }
    if (!source_git_blob_oid_bytes(blob.bytes, blob.byte_count, retained_oid) ||
        strcmp(retained_oid, blob.identity.expected_git_blob_oid) != 0) {
        yvex_source_metadata_blob_release(&blob);
        return provenance_refuse(err, YVEX_ERR_FORMAT, "source_metadata_read",
            "retained metadata bytes differ from provider identity");
    }
    memset(&after, 0, sizeof(after));
    rc = source_metadata_identity(verification, canonical_name, &after, err);
    if (rc != YVEX_OK || after.file_bytes != blob.identity.file_bytes ||
        strcmp(after.observed_git_blob_oid, blob.identity.observed_git_blob_oid) != 0) {
        yvex_source_metadata_blob_release(&blob);
        if (rc == YVEX_OK)
            yvex_error_set(err,
                           YVEX_ERR_FORMAT,
                           "source_metadata_read",
                           "metadata sidecar identity drifted during read");
        return rc == YVEX_OK ? YVEX_ERR_FORMAT : rc;
    }
    *out = blob;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Releases one metadata blob and resets every observable fact.
 *
 * Releases only resources owned by source provenance; cleanup remains deterministic.
 */
void yvex_source_metadata_blob_release(yvex_source_metadata_blob *blob) {
    if (!blob)
        return;
    free(blob->bytes);
    memset(blob, 0, sizeof(*blob));
}

int yvex_source_provenance_verify_file(const yvex_source_verify_options *options,
                                       const char *name,
                                       int verify_upstream_index,
                                       yvex_source_verification *out,
                                       yvex_error *err) {
    char revision[128];
    char oid[128];
    char path[YVEX_PATH_CAP];
    unsigned long long size;
    int rc;

    if (!options || !name || !out ||
        !source_metadata_read(
            options->source_path, name, revision, sizeof(revision), oid, sizeof(oid))) {
        yvex_source_verification_add_blocker(out, "missing-source-revision");
        return YVEX_OK;
    }
    if (!out->revision[0]) {
        snprintf(out->revision, sizeof(out->revision), "%s", revision);
    } else if (strcmp(out->revision, revision) != 0) {
        yvex_source_verification_add_blocker(out, "inconsistent-source-revision");
    }
    if (strcmp(revision, options->identity->upstream_revision) != 0) {
        yvex_source_verification_add_blocker(out, "stale-source-revision");
    }
    if (!verify_upstream_index)
        return YVEX_OK;
    if (!source_oid_is_sha1(oid)) {
        yvex_source_verification_add_blocker(out, "upstream-index-identity-mismatch");
        return YVEX_OK;
    }
    memcpy(out->upstream_index_oid, oid, 41u);
    if (!yvex_source_path_join(path, sizeof(path), options->source_path, name) ||
        !yvex_source_regular_file(path, &size) || size != options->identity->upstream_index_size ||
        strcmp(oid, options->identity->upstream_index_oid) != 0) {
        yvex_source_verification_add_blocker(out, "upstream-index-identity-mismatch");
        return YVEX_OK;
    }
    rc = yvex_source_git_blob_oid_file(path, out->local_index_oid, err);
    if (rc != YVEX_OK)
        return rc;
    if (strcmp(out->local_index_oid, oid) != 0) {
        yvex_source_verification_add_blocker(out, "upstream-index-identity-mismatch");
    } else {
        out->upstream_index_identity_verified = 1;
    }
    return YVEX_OK;
}

void yvex_source_provenance_finalize(const yvex_source_verify_options *options,
                                     yvex_source_verification *out) {
    size_t i;
    int manifest_ref_valid = 1;

    if (!options || !out)
        return;
    if (!out->manifest_revision[0]) {
        yvex_source_verification_add_blocker(out, "missing-source-revision");
        manifest_ref_valid = 0;
    } else if (strcmp(out->manifest_revision, "unknown") == 0 ||
               strcmp(out->manifest_revision, "unverified") == 0) {
        yvex_source_verification_add_blocker(out, "unverifiable-source-revision");
        manifest_ref_valid = 0;
    } else {
        for (i = 0u; out->manifest_revision[i]; ++i) {
            unsigned char ch = (unsigned char)out->manifest_revision[i];
            if (!isalnum(ch) && ch != '.' && ch != '_' && ch != '-' && ch != '/') {
                yvex_source_verification_add_blocker(out, "unverifiable-source-revision");
                manifest_ref_valid = 0;
                break;
            }
        }
    }
    if (!out->revision[0] || strcmp(out->revision, options->identity->upstream_revision) != 0 ||
        yvex_source_verification_has_blocker(out, "missing-source-revision") ||
        yvex_source_verification_has_blocker(out, "inconsistent-source-revision") ||
        yvex_source_verification_has_blocker(out, "stale-source-revision") || !manifest_ref_valid) {
        out->revision_verified = 0;
        return;
    }
    if (out->manifest_revision[0] && yvex_source_revision_is_commit(out->manifest_revision) &&
        strcmp(out->manifest_revision, out->revision) != 0) {
        yvex_source_verification_add_blocker(out, "inconsistent-source-revision");
        return;
    }
    out->revision_verified = 1;
}

int yvex_source_provenance_manifest_matches(const yvex_source_verify_options *options,
                                            const yvex_source_verification *out) {
    int schema_v2;
    int schema_v3;
    int common;

    if (!options || !out)
        return 0;
    schema_v2 = strcmp(out->manifest_schema, "yvex.source_manifest.v2") == 0;
    schema_v3 = strcmp(out->manifest_schema, "yvex.source_manifest.v3") == 0;
    common =
        (schema_v2 || schema_v3) && strcmp(out->manifest_status, "complete") == 0 &&
        strcmp(out->manifest_target_id, options->identity->target_id) == 0 &&
        strcmp(out->repository_id, options->identity->upstream_repo_id) == 0 &&
        strcmp(out->manifest_revision, out->revision) == 0 &&
        strcmp(out->inventory_authority, options->identity->upstream_inventory_authority) == 0 &&
        strcmp(out->manifest_config_status, "verified") == 0 &&
        strcmp(out->manifest_tokenizer_status, "verified") == 0 &&
        strcmp(out->upstream_index_oid,
               options->identity->upstream_index_oid ? options->identity->upstream_index_oid
                                                     : "not-applicable") == 0 &&
        out->manifest_source_file_count == out->source_file_count &&
        out->manifest_source_total_bytes == out->source_total_bytes &&
        out->manifest_shard_count == out->shard_count &&
        out->manifest_shard_bytes == out->shard_bytes &&
        out->manifest_header_tensor_count == out->header_tensor_count;
    if (!common)
        return 0;
    if (schema_v2) {
        return strcmp(out->verification_stage, "exact-source-metadata-header-verified") == 0 &&
               strcmp(out->manifest_payload_digest_status, "not-verified") == 0;
    }
    return strcmp(out->verification_stage, "exact-source-payload-verified") == 0 &&
           strcmp(out->manifest_payload_digest_status, out->manifest_payload_trust_class) == 0 &&
           out->manifest_payload_shard_count == out->shard_count &&
           out->manifest_payload_bytes == out->shard_bytes &&
           out->manifest_payload_source_snapshot_identity == out->source_snapshot_identity &&
           out->manifest_payload_tensor_count == out->header_tensor_count &&
           out->manifest_payload_logical_tensor_bytes == out->declared_tensor_bytes;
}

static const char *const source_acquisition_failure_names[] = {
    "none", "invalid-argument", "manifest-missing", "manifest-format",
    "source-identity", "resource-budget", "path", "symlink", "non-regular",
    "size", "digest", "duplicate", "io", "allocation"
};

static const char *source_acquisition_failure_name(
    yvex_source_acquisition_failure_code code)
{
    size_t count = sizeof(source_acquisition_failure_names) /
                   sizeof(source_acquisition_failure_names[0]);

    return (unsigned int)code < count ? source_acquisition_failure_names[code] : "unknown";
}

static int source_acquisition_refuse(
    yvex_source_acquisition_failure *failure,
    yvex_source_acquisition_failure_code code,
    unsigned long long file_index,
    const char *path,
    yvex_status status,
    yvex_error *err,
    const char *message)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->file_index = file_index;
        yvex_core_text_copy(failure->path, sizeof(failure->path), path ? path : "");
    }
    yvex_error_setf(err, status, "source_acquisition", "%s: %s",
                    source_acquisition_failure_name(code), message);
    return status;
}

void yvex_source_acquisition_options_default(yvex_source_acquisition_options *options)
{
    if (!options) return;
    memset(options, 0, sizeof(*options));
    options->maximum_files = YVEX_SOURCE_ACQUISITION_FILE_CAP;
    options->maximum_source_bytes = 200000000000ull;
    options->verify_digests = 1;
}

static int source_acquisition_path_valid(const char *path)
{
    const char *component = path;
    const char *cursor;

    if (!path || !path[0] || path[0] == '/') return 0;
    for (cursor = path;; ++cursor) {
        unsigned char byte = (unsigned char)*cursor;
        size_t length;

        if (byte == '\\' || byte == '\n' || byte == '\r' || byte == '\t') return 0;
        if (byte != '/' && byte != '\0') continue;
        length = (size_t)(cursor - component);
        if (length == 0u || (length == 1u && component[0] == '.') ||
            (length == 2u && component[0] == '.' && component[1] == '.')) return 0;
        if (byte == '\0') return 1;
        component = cursor + 1;
    }
}

static int source_acquisition_parse_class(yvex_json *json,
                                          yvex_source_acquisition_file *file)
{
    char value[16];

    if (!yvex_json_string(json, value, sizeof(value))) return 0;
    if (strcmp(value, "metadata") == 0) {
        file->classification = YVEX_SOURCE_ACQUISITION_FILE_METADATA;
        return 1;
    }
    if (strcmp(value, "shard") == 0) {
        file->classification = YVEX_SOURCE_ACQUISITION_FILE_SHARD;
        return 1;
    }
    return 0;
}

static int source_acquisition_parse_file(yvex_json *json,
                                         yvex_source_acquisition_file *file)
{
    enum {
        FIELD_ACTUAL_SHA = 1u << 0, FIELD_ACTUAL_SIZE = 1u << 1,
        FIELD_CLASS = 1u << 2, FIELD_COMPONENT = 1u << 3,
        FIELD_EXPECTED_SHA = 1u << 4, FIELD_EXPECTED_SIZE = 1u << 5,
        FIELD_GIT = 1u << 6, FIELD_LFS = 1u << 7, FIELD_PATH = 1u << 8,
        FIELD_VERIFIED = 1u << 9, FIELD_XET = 1u << 10
    };
    const unsigned int required = (1u << 11) - 1u;
    yvex_json_iter iterator;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];
    unsigned int seen = 0u;
    int verified = 0;

    memset(file, 0, sizeof(*file));
    if (!yvex_json_iter_begin(json, &iterator, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iterator, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
#define SOURCE_ACQUISITION_TEXT(name, bit, member)                                    \
        if (strcmp(key, name) == 0) {                                                 \
            if ((seen & bit) || !yvex_json_string(                                   \
                                    json, file->member, sizeof(file->member))) return 0; \
            seen |= bit;                                                              \
        }
        SOURCE_ACQUISITION_TEXT("actual_sha256", FIELD_ACTUAL_SHA, actual_sha256)
        else if (strcmp(key, "actual_size") == 0) {
            if ((seen & FIELD_ACTUAL_SIZE) || !yvex_json_u64(json, &file->actual_size)) return 0;
            seen |= FIELD_ACTUAL_SIZE;
        }
        else if (strcmp(key, "classification") == 0) {
            if ((seen & FIELD_CLASS) || !source_acquisition_parse_class(json, file)) return 0;
            seen |= FIELD_CLASS;
        }
        else SOURCE_ACQUISITION_TEXT("component", FIELD_COMPONENT, component)
        else SOURCE_ACQUISITION_TEXT("expected_sha256", FIELD_EXPECTED_SHA, expected_sha256)
        else if (strcmp(key, "expected_size") == 0) {
            if ((seen & FIELD_EXPECTED_SIZE) || !yvex_json_u64(json, &file->expected_size)) return 0;
            seen |= FIELD_EXPECTED_SIZE;
        }
        else SOURCE_ACQUISITION_TEXT("git_oid", FIELD_GIT, git_oid)
        else SOURCE_ACQUISITION_TEXT("lfs_oid", FIELD_LFS, lfs_oid)
        else SOURCE_ACQUISITION_TEXT("path", FIELD_PATH, path)
        else if (strcmp(key, "verified") == 0) {
            if ((seen & FIELD_VERIFIED) || !yvex_json_bool(json, &verified)) return 0;
            seen |= FIELD_VERIFIED;
        }
        else SOURCE_ACQUISITION_TEXT("xet_hash", FIELD_XET, xet_hash)
        else if (!yvex_json_skip_value(json)) return 0;
#undef SOURCE_ACQUISITION_TEXT
    }
    return item == YVEX_JSON_ITEM_END && seen == required && verified &&
           source_acquisition_path_valid(file->path) && file->component[0] &&
           file->actual_size == file->expected_size &&
           yvex_sha256_hex_valid(file->actual_sha256) &&
           (!file->expected_sha256[0] ||
            (yvex_sha256_hex_valid(file->expected_sha256) &&
             strcmp(file->actual_sha256, file->expected_sha256) == 0));
}

static int source_acquisition_parse_files(yvex_json *json,
                                          yvex_source_acquisition *acquisition,
                                          unsigned long long maximum_files)
{
    yvex_json_iter iterator;
    yvex_json_item item;
    unsigned long long count = 0u;

    acquisition->files = (yvex_source_acquisition_file *)calloc(
        (size_t)maximum_files, sizeof(acquisition->files[0]));
    if (!acquisition->files) return -1;
    if (!yvex_json_iter_begin(json, &iterator, YVEX_JSON_COLLECTION_ARRAY)) return 0;
    while ((item = yvex_json_array_value(&iterator)) == YVEX_JSON_ITEM_READY) {
        yvex_source_acquisition_file *file;
        if (count >= maximum_files) return -2;
        file = &acquisition->files[count];
        if (!source_acquisition_parse_file(json, file)) return 0;
        if (count && strcmp(acquisition->files[count - 1u].path, file->path) >= 0) return -3;
        if (file->classification == YVEX_SOURCE_ACQUISITION_FILE_SHARD) {
            acquisition->facts.shard_count++;
            if (!yvex_core_u64_add(acquisition->facts.shard_bytes, file->actual_size,
                                   &acquisition->facts.shard_bytes)) return -2;
        } else if (!yvex_core_u64_add(acquisition->facts.metadata_bytes, file->actual_size,
                                      &acquisition->facts.metadata_bytes)) return -2;
        if (!yvex_core_u64_add(acquisition->facts.source_bytes, file->actual_size,
                               &acquisition->facts.source_bytes)) return -2;
        count++;
    }
    acquisition->facts.file_count = count;
    return item == YVEX_JSON_ITEM_END && !iterator.trailing_separator && count ? 1 : 0;
}

typedef struct {
    unsigned long long file_count;
    unsigned long long shard_count;
    unsigned long long metadata_bytes;
    unsigned long long shard_bytes;
    unsigned long long source_bytes;
    unsigned int seen;
    int complete;
} source_acquisition_declared;

static int source_acquisition_parse_manifest(
    const char *data,
    size_t length,
    yvex_source_acquisition *acquisition,
    source_acquisition_declared *declared,
    unsigned long long maximum_files)
{
    yvex_json json;
    yvex_json_iter iterator;
    yvex_json_item item;
    char key[YVEX_JSON_KEY_CAP];
    char schema[64] = {0};

    yvex_json_init(&json, data, length);
    if (!yvex_json_iter_begin(&json, &iterator, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iterator, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "acquisition_complete") == 0) {
            if (!yvex_json_bool(&json, &declared->complete)) return 0;
            declared->seen |= 1u << 0;
        } else if (strcmp(key, "acquisition_identity") == 0) {
            if (!yvex_json_string(&json, acquisition->facts.acquisition_identity,
                                  sizeof(acquisition->facts.acquisition_identity))) return 0;
            declared->seen |= 1u << 1;
        } else if (strcmp(key, "admitted_subtree") == 0) {
            if (!yvex_json_string(&json, acquisition->facts.subtree,
                                  sizeof(acquisition->facts.subtree))) return 0;
            declared->seen |= 1u << 2;
        } else if (strcmp(key, "files") == 0) {
            int parsed = source_acquisition_parse_files(&json, acquisition, maximum_files);
            if (parsed != 1) return parsed;
            declared->file_count = acquisition->facts.file_count;
            declared->seen |= 1u << 3;
        } else if (strcmp(key, "metadata_bytes") == 0) {
            if (!yvex_json_u64(&json, &declared->metadata_bytes)) return 0;
            declared->seen |= 1u << 4;
        } else if (strcmp(key, "repository") == 0) {
            if (!yvex_json_string(&json, acquisition->facts.repository,
                                  sizeof(acquisition->facts.repository))) return 0;
            declared->seen |= 1u << 5;
        } else if (strcmp(key, "revision") == 0) {
            if (!yvex_json_string(&json, acquisition->facts.revision,
                                  sizeof(acquisition->facts.revision))) return 0;
            declared->seen |= 1u << 6;
        } else if (strcmp(key, "schema") == 0) {
            if (!yvex_json_string(&json, schema, sizeof(schema))) return 0;
            declared->seen |= 1u << 7;
        } else if (strcmp(key, "shard_bytes") == 0) {
            if (!yvex_json_u64(&json, &declared->shard_bytes)) return 0;
            declared->seen |= 1u << 8;
        } else if (strcmp(key, "shards") == 0) {
            if (!yvex_json_u64(&json, &declared->shard_count)) return 0;
            declared->seen |= 1u << 9;
        } else if (strcmp(key, "source_bytes") == 0) {
            if (!yvex_json_u64(&json, &declared->source_bytes)) return 0;
            declared->seen |= 1u << 10;
        } else if (!yvex_json_skip_value(&json)) return 0;
    }
    return item == YVEX_JSON_ITEM_END && yvex_json_complete(&json) &&
           declared->seen == ((1u << 11) - 1u) && declared->complete &&
           strcmp(schema, YVEX_SOURCE_ACQUISITION_SCHEMA) == 0;
}

static int source_acquisition_open_relative(int root_fd,
                                            const char *path,
                                            int *symlink_refused)
{
    char copy[YVEX_PATH_CAP];
    char *cursor;
    char *save = NULL;
    int directory_fd;

    if (!source_acquisition_path_valid(path) || strlen(path) >= sizeof(copy)) return -1;
    memcpy(copy, path, strlen(path) + 1u);
    directory_fd = dup(root_fd);
    if (directory_fd < 0) return -1;
    cursor = strtok_r(copy, "/", &save);
    while (cursor) {
        char *next = strtok_r(NULL, "/", &save);
        int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
        int next_fd;

        if (next) flags |= O_DIRECTORY;
        next_fd = openat(directory_fd, cursor, flags);
        close(directory_fd);
        if (next_fd < 0) {
            if (symlink_refused && errno == ELOOP) *symlink_refused = 1;
            return -1;
        }
        directory_fd = next_fd;
        cursor = next;
    }
    return directory_fd;
}

static int source_acquisition_hash_file(
    int root_fd,
    const yvex_source_acquisition_file *file,
    unsigned long long *bytes_read,
    char digest_hex[65],
    int *symlink_refused)
{
    unsigned char buffer[1024u * 1024u];
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    struct stat before;
    struct stat after;
    unsigned long long total = 0u;
    int fd = source_acquisition_open_relative(root_fd, file->path, symlink_refused);
    ssize_t count;

    if (fd < 0) return 0;
    if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size < 0 ||
        (unsigned long long)before.st_size != file->expected_size) {
        close(fd);
        return 0;
    }
    yvex_sha256_init(&hash);
    while ((count = read(fd, buffer, sizeof(buffer))) > 0) {
        if (!yvex_sha256_update(&hash, buffer, (size_t)count) ||
            !yvex_core_u64_add(total, (unsigned long long)count, &total)) {
            close(fd);
            return 0;
        }
    }
    if (count < 0 || fstat(fd, &after) != 0 || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        before.st_mtime != after.st_mtime || total != file->expected_size ||
        !yvex_sha256_final(&hash, digest)) {
        close(fd);
        return 0;
    }
    close(fd);
    yvex_sha256_hex(digest, digest_hex);
    *bytes_read = total;
    return 1;
}

static int source_acquisition_identity(const yvex_source_acquisition *acquisition,
                                       char output[65])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, YVEX_SOURCE_ACQUISITION_SCHEMA) ||
        !yvex_sha256_update_text(&hash, acquisition->facts.repository) ||
        !yvex_sha256_update_text(&hash, acquisition->facts.revision) ||
        !yvex_sha256_update_text(&hash, acquisition->facts.subtree) ||
        !yvex_sha256_update_u64(&hash, acquisition->facts.file_count)) return 0;
    for (index = 0u; index < acquisition->facts.file_count; ++index) {
        const yvex_source_acquisition_file *file = &acquisition->files[index];
        const char *classification =
            file->classification == YVEX_SOURCE_ACQUISITION_FILE_SHARD ? "shard" : "metadata";
        if (!yvex_sha256_update_text(&hash, file->path) ||
            !yvex_sha256_update_u64(&hash, file->expected_size) ||
            !yvex_sha256_update_text(&hash, file->expected_sha256) ||
            !yvex_sha256_update_text(&hash, file->actual_sha256) ||
            !yvex_sha256_update_text(&hash, file->git_oid) ||
            !yvex_sha256_update_text(&hash, file->lfs_oid) ||
            !yvex_sha256_update_text(&hash, file->xet_hash) ||
            !yvex_sha256_update_text(&hash, classification) ||
            !yvex_sha256_update_text(&hash, file->component)) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int source_acquisition_verify_files(
    yvex_source_acquisition *acquisition,
    const yvex_source_acquisition_options *options,
    yvex_source_acquisition_failure *failure,
    yvex_error *err)
{
    struct stat root_status;
    unsigned long long index;
    int root_fd;

    if (lstat(options->source_root, &root_status) != 0 ||
        !S_ISDIR(root_status.st_mode) || S_ISLNK(root_status.st_mode)) {
        return source_acquisition_refuse(
            failure, YVEX_SOURCE_ACQUISITION_FAILURE_PATH, 0u,
            options->source_root, YVEX_ERR_IO, err,
            "source root must be a real directory");
    }
    root_fd = open(options->source_root,
                   O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
    if (root_fd < 0) {
        return source_acquisition_refuse(
            failure, YVEX_SOURCE_ACQUISITION_FAILURE_IO, 0u,
            options->source_root, YVEX_ERR_IO, err,
            "cannot open source root");
    }
    for (index = 0u; index < acquisition->facts.file_count; ++index) {
        const yvex_source_acquisition_file *file = &acquisition->files[index];
        unsigned long long bytes = 0u;
        char digest[65];
        int symlink_refused = 0;

        if (!source_acquisition_hash_file(root_fd, file, &bytes, digest,
                                          &symlink_refused)) {
            close(root_fd);
            return source_acquisition_refuse(
                failure,
                symlink_refused ? YVEX_SOURCE_ACQUISITION_FAILURE_SYMLINK
                                : YVEX_SOURCE_ACQUISITION_FAILURE_SIZE,
                index, file->path, YVEX_ERR_FORMAT, err,
                symlink_refused
                    ? "source path contains a symlink"
                    : "source file is missing, non-regular, replaced, or wrong-sized");
        }
        if (options->verify_digests &&
            strcmp(digest, file->actual_sha256) != 0) {
            close(root_fd);
            return source_acquisition_refuse(
                failure, YVEX_SOURCE_ACQUISITION_FAILURE_DIGEST, index,
                file->path, YVEX_ERR_FORMAT, err,
                "source file digest mismatch");
        }
        if (!yvex_core_u64_add(acquisition->facts.payload_bytes_read, bytes,
                               &acquisition->facts.payload_bytes_read)) {
            close(root_fd);
            return source_acquisition_refuse(
                failure, YVEX_SOURCE_ACQUISITION_FAILURE_RESOURCE_BUDGET,
                index, file->path, YVEX_ERR_BOUNDS, err,
                "source verification byte count overflow");
        }
    }
    close(root_fd);
    yvex_core_execution_observation_record(
        YVEX_CORE_OBSERVE_SOURCE_PAYLOAD_BYTES,
        acquisition->facts.payload_bytes_read);
    return YVEX_OK;
}

static int source_acquisition_manifest_path(
    const yvex_source_acquisition_options *options,
    char output[YVEX_PATH_CAP],
    yvex_source_acquisition_failure *failure,
    yvex_error *err)
{
    if (yvex_source_path_join(output, YVEX_PATH_CAP, options->source_root,
                              YVEX_SOURCE_ACQUISITION_MANIFEST)) return YVEX_OK;
    return source_acquisition_refuse(
        failure, YVEX_SOURCE_ACQUISITION_FAILURE_PATH, 0u, NULL,
        YVEX_ERR_BOUNDS, err, "source manifest path exceeds bounds");
}

static int source_acquisition_validate_manifest(
    yvex_source_acquisition *acquisition,
    const source_acquisition_declared *declared,
    const yvex_source_acquisition_options *options,
    yvex_source_acquisition_failure *failure,
    yvex_error *err)
{
    char computed_identity[65];

    if (strcmp(acquisition->facts.repository, options->expected_repository) != 0 ||
        strcmp(acquisition->facts.revision, options->expected_revision) != 0 ||
        strcmp(acquisition->facts.subtree, options->expected_subtree) != 0) {
        return source_acquisition_refuse(
            failure, YVEX_SOURCE_ACQUISITION_FAILURE_SOURCE_IDENTITY,
            0u, NULL, YVEX_ERR_FORMAT, err,
            "repository, revision, or subtree identity mismatch");
    }
    if (acquisition->facts.source_bytes > options->maximum_source_bytes ||
        declared->file_count != acquisition->facts.file_count ||
        declared->shard_count != acquisition->facts.shard_count ||
        declared->metadata_bytes != acquisition->facts.metadata_bytes ||
        declared->shard_bytes != acquisition->facts.shard_bytes ||
        declared->source_bytes != acquisition->facts.source_bytes) {
        return source_acquisition_refuse(
            failure, YVEX_SOURCE_ACQUISITION_FAILURE_RESOURCE_BUDGET,
            0u, NULL, YVEX_ERR_BOUNDS, err,
            "source acquisition totals or budget disagree");
    }
    if (!source_acquisition_identity(acquisition, computed_identity) ||
        strcmp(computed_identity, acquisition->facts.acquisition_identity) != 0) {
        return source_acquisition_refuse(
            failure, YVEX_SOURCE_ACQUISITION_FAILURE_SOURCE_IDENTITY,
            0u, NULL, YVEX_ERR_FORMAT, err,
            "source acquisition identity mismatch");
    }
    return YVEX_OK;
}

int yvex_source_acquisition_open(
    yvex_source_acquisition **out,
    const yvex_source_acquisition_options *options,
    yvex_source_acquisition_failure *failure,
    yvex_error *err)
{
    yvex_source_acquisition *acquisition = NULL;
    source_acquisition_declared declared = {0};
    char *manifest = NULL;
    char manifest_path[YVEX_PATH_CAP];
    size_t manifest_length = 0u;
    int parsed;
    int rc;

    if (!out || !options || !options->source_root ||
        !options->expected_repository || !options->expected_revision ||
        !options->expected_subtree || !options->maximum_files ||
        options->maximum_files > YVEX_SOURCE_ACQUISITION_FILE_CAP ||
        !options->maximum_source_bytes) {
        return source_acquisition_refuse(
            failure, YVEX_SOURCE_ACQUISITION_FAILURE_INVALID_ARGUMENT,
            0u, NULL, YVEX_ERR_INVALID_ARG, err,
            "bounded source acquisition options are required");
    }
    *out = NULL;
    rc = source_acquisition_manifest_path(options, manifest_path, failure, err);
    if (rc != YVEX_OK) return rc;
    manifest = yvex_read_bounded_file(
        manifest_path, SOURCE_ACQUISITION_MANIFEST_CAP, &manifest_length, err);
    if (!manifest) {
        return source_acquisition_refuse(
            failure, YVEX_SOURCE_ACQUISITION_FAILURE_MANIFEST_MISSING,
            0u, YVEX_SOURCE_ACQUISITION_MANIFEST, YVEX_ERR_IO, err,
            "source acquisition manifest is unavailable");
    }
    acquisition = (yvex_source_acquisition *)calloc(1u, sizeof(*acquisition));
    if (!acquisition) {
        free(manifest);
        return source_acquisition_refuse(
            failure, YVEX_SOURCE_ACQUISITION_FAILURE_ALLOCATION,
            0u, NULL, YVEX_ERR_NOMEM, err,
            "source acquisition allocation failed");
    }
    parsed = source_acquisition_parse_manifest(
        manifest, manifest_length, acquisition, &declared,
        options->maximum_files);
    free(manifest);
    if (parsed != 1) {
        yvex_source_acquisition_failure_code code =
            parsed == -1 ? YVEX_SOURCE_ACQUISITION_FAILURE_ALLOCATION
                         : parsed == -2
                               ? YVEX_SOURCE_ACQUISITION_FAILURE_RESOURCE_BUDGET
                               : parsed == -3
                                     ? YVEX_SOURCE_ACQUISITION_FAILURE_DUPLICATE
                                     : YVEX_SOURCE_ACQUISITION_FAILURE_MANIFEST_FORMAT;
        rc = source_acquisition_refuse(
            failure, code, 0u, YVEX_SOURCE_ACQUISITION_MANIFEST,
            parsed == -1 ? YVEX_ERR_NOMEM : YVEX_ERR_FORMAT, err,
            "source acquisition manifest is not canonical");
        yvex_source_acquisition_release(&acquisition);
        return rc;
    }
    rc = source_acquisition_validate_manifest(
        acquisition, &declared, options, failure, err);
    if (rc == YVEX_OK) {
        rc = source_acquisition_verify_files(
            acquisition, options, failure, err);
    }
    if (rc != YVEX_OK) {
        yvex_source_acquisition_release(&acquisition);
        return rc;
    }
    acquisition->facts.complete = 1;
    if (failure) memset(failure, 0, sizeof(*failure));
    *out = acquisition;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_source_acquisition_release(yvex_source_acquisition **acquisition)
{
    if (!acquisition || !*acquisition) return;
    free((*acquisition)->files);
    memset(*acquisition, 0, sizeof(**acquisition));
    free(*acquisition);
    *acquisition = NULL;
}

const yvex_source_acquisition_facts *yvex_source_acquisition_facts_get(
    const yvex_source_acquisition *acquisition)
{
    return acquisition ? &acquisition->facts : NULL;
}

const yvex_source_acquisition_file *yvex_source_acquisition_file_at(
    const yvex_source_acquisition *acquisition,
    unsigned long long index)
{
    return acquisition && index < acquisition->facts.file_count
               ? &acquisition->files[index]
               : NULL;
}
