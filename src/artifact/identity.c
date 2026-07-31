/*
 * Compute canonical identities over exact artifact byte sequences.
 *
 * The identity binds every byte and exact length; partial reads never publish. Physical identity
 * does not prove semantic completeness or support.
 */
#include <ctype.h>
#include <dlfcn.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/artifact.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/core.h>
typedef void *(*artifact_evp_context_new_fn)(void);
typedef void (*artifact_evp_context_free_fn)(void *context);
typedef const void *(*artifact_evp_sha256_fn)(void);
typedef int (*artifact_evp_digest_init_fn)(void *context, const void *algorithm, void *engine);
typedef int (*artifact_evp_digest_update_fn)(void *context, const void *bytes, size_t byte_count);
typedef int (*artifact_evp_digest_final_fn)(void *context, unsigned char *digest,
                                            unsigned int *digest_bytes);
typedef struct {
    void *library;
    artifact_evp_context_new_fn context_new;
    artifact_evp_context_free_fn context_free;
    artifact_evp_sha256_fn sha256;
    artifact_evp_digest_init_fn digest_init;
    artifact_evp_digest_update_fn digest_update;
    artifact_evp_digest_final_fn digest_final;
    int available;
} artifact_hash_provider;
typedef struct {
    const char *name;
    size_t offset;
    size_t size;
} artifact_hash_symbol_spec;
static const artifact_hash_symbol_spec artifact_hash_symbols[] = {
    {"EVP_MD_CTX_new", offsetof(artifact_hash_provider, context_new), sizeof(artifact_evp_context_new_fn)},
    {"EVP_MD_CTX_free", offsetof(artifact_hash_provider, context_free), sizeof(artifact_evp_context_free_fn)},
    {"EVP_sha256", offsetof(artifact_hash_provider, sha256), sizeof(artifact_evp_sha256_fn)},
    {"EVP_DigestInit_ex", offsetof(artifact_hash_provider, digest_init), sizeof(artifact_evp_digest_init_fn)},
    {"EVP_DigestUpdate", offsetof(artifact_hash_provider, digest_update), sizeof(artifact_evp_digest_update_fn)},
    {"EVP_DigestFinal_ex", offsetof(artifact_hash_provider, digest_final), sizeof(artifact_evp_digest_final_fn)},
};
typedef struct {
    const artifact_hash_provider *provider;
    void *context;
    yvex_sha256 portable;
    int accelerated;
} artifact_hash;

static int artifact_hash_symbol(void *library, const char *name, void *destination,
                                size_t destination_bytes)
{
    void *symbol;
    if (!library || !name || !destination || destination_bytes != sizeof(symbol))
        return 0;
    symbol = dlsym(library, name);
    if (!symbol)
        return 0;
    memcpy(destination, &symbol, sizeof(symbol));
    return 1;
}

static void artifact_hash_provider_open(artifact_hash_provider *provider)
{
    static const char *const libraries[] = {"libcrypto.so.3", "libcrypto.so"};
    size_t index;
    memset(provider, 0, sizeof(*provider));
    for (index = 0u; index < sizeof(libraries) / sizeof(libraries[0]); ++index) {
        provider->library = dlopen(libraries[index], RTLD_NOW | RTLD_LOCAL);
        if (provider->library)
            break;
    }
    if (!provider->library)
        return;
    provider->available = 1;
    for (index = 0u; index < sizeof(artifact_hash_symbols) / sizeof(artifact_hash_symbols[0]); ++index)
        if (!artifact_hash_symbol(provider->library, artifact_hash_symbols[index].name,
                                  (unsigned char *)provider + artifact_hash_symbols[index].offset,
                                  artifact_hash_symbols[index].size)) {
            provider->available = 0;
            break;
        }
    if (!provider->available) {
        dlclose(provider->library);
        memset(provider, 0, sizeof(*provider));
    }
}

static void artifact_hash_provider_close(artifact_hash_provider *provider)
{
    if (!provider)
        return;
    if (provider->library)
        dlclose(provider->library);
    memset(provider, 0, sizeof(*provider));
}

static int artifact_hash_init(artifact_hash *hash, const artifact_hash_provider *provider)
{
    memset(hash, 0, sizeof(*hash));
    hash->provider = provider;
    if (provider && provider->available) {
        hash->context = provider->context_new();
        if (hash->context && provider->digest_init(hash->context, provider->sha256(), NULL) == 1) {
            hash->accelerated = 1;
            return 1;
        }
        if (hash->context)
            provider->context_free(hash->context);
        hash->context = NULL;
    }
    yvex_sha256_init(&hash->portable);
    return 1;
}

static int artifact_hash_update(artifact_hash *hash, const void *bytes, size_t byte_count)
{
    if (!hash || (!bytes && byte_count))
        return 0;
    if (hash->accelerated)
        return hash->provider->digest_update(hash->context, bytes, byte_count) == 1;
    return yvex_sha256_update(&hash->portable, bytes, byte_count);
}

static int artifact_hash_final(artifact_hash *hash,
                               unsigned char digest[YVEX_SHA256_DIGEST_BYTES])
{
    unsigned int digest_bytes = 0u;
    int ok;
    if (!hash || !digest)
        return 0;
    if (!hash->accelerated)
        return yvex_sha256_final(&hash->portable, digest);
    ok = hash->provider->digest_final(hash->context, digest, &digest_bytes) == 1 &&
         digest_bytes == YVEX_SHA256_DIGEST_BYTES;
    hash->provider->context_free(hash->context);
    hash->context = NULL;
    hash->accelerated = 0;
    return ok;
}

static void artifact_hash_abort(artifact_hash *hash)
{
    if (hash && hash->accelerated && hash->context)
        hash->provider->context_free(hash->context);
    if (hash)
        memset(hash, 0, sizeof(*hash));
}
/*
 * Hash an entire named file through the canonical portable path contract.
 *
 * Performs bounded sequential reads and publishes size only after full coverage.
 */
static int artifact_hash_path(const char *path, unsigned int error_kind,
                              unsigned char digest[YVEX_SHA256_DIGEST_BYTES],
                              unsigned long long *file_size, yvex_error *err)
{
    const char *context = error_kind ? "yvex_artifact_identity_read"
                                     : "yvex_artifact_compute_sha256";
    const char *purpose = error_kind ? "identity" : "sha256";
    unsigned char buffer[65536];
    yvex_sha256 hash;
    unsigned long long size = 0u;
    FILE *stream = fopen(path, "rb");
    int rc = YVEX_OK;
    if (!stream) {
        yvex_error_setf(err, YVEX_ERR_IO, context, "cannot open artifact for %s: %s", purpose,
                        path);
        return YVEX_ERR_IO;
    }
    yvex_sha256_init(&hash);
    for (;;) {
        size_t got = fread(buffer, 1u, sizeof(buffer), stream);
        if (got && !yvex_sha256_update(&hash, buffer, got)) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, context, "artifact SHA-256 length overflow");
            rc = YVEX_ERR_BOUNDS;
            break;
        }
        if (file_size && ULLONG_MAX - size < (unsigned long long)got) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, context, "artifact file size overflow");
            rc = YVEX_ERR_BOUNDS;
            break;
        }
        size += (unsigned long long)got;
        if (got == sizeof(buffer))
            continue;
        if (ferror(stream)) {
            yvex_error_setf(err, YVEX_ERR_IO, context, "cannot read artifact for %s: %s", purpose,
                            path);
            rc = YVEX_ERR_IO;
        }
        break;
    }
    (void)fclose(stream);
    if (rc == YVEX_OK && !yvex_sha256_final(&hash, digest)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, context, "artifact SHA-256 finalization failed");
        rc = YVEX_ERR_BOUNDS;
    }
    if (rc == YVEX_OK && file_size)
        *file_size = size;
    return rc;
}

static int artifact_hash_range(const yvex_artifact *artifact, unsigned long long start,
                               unsigned long long byte_count, unsigned char *buffer,
                               size_t buffer_bytes, artifact_hash *hash,
                               int (*progress)(void *, unsigned long long,
                                               unsigned long long),
                               void *progress_context, yvex_error *err)
{
    unsigned long long delivered = 0u;
    while (delivered < byte_count) {
        unsigned long long remaining = byte_count - delivered;
        unsigned long long offset;
        size_t take = remaining > buffer_bytes ? buffer_bytes : (size_t)remaining;
        if (!yvex_core_u64_add(start, delivered, &offset))
            return YVEX_ERR_BOUNDS;
        int rc = yvex_artifact_read_at(artifact, offset, buffer, take, err);
        if (rc != YVEX_OK)
            return rc;
        if (!artifact_hash_update(hash, buffer, take))
            return YVEX_ERR_BOUNDS;
        delivered += (unsigned long long)take;
        if (progress && !progress(progress_context, delivered, byte_count)) {
            yvex_error_set(err, YVEX_ERR_CANCELLED,
                           "yvex_artifact_identity_read_open",
                           "artifact identity read was cancelled");
            return YVEX_ERR_CANCELLED;
        }
    }
    return YVEX_OK;
}
/*
 * Compare every stable artifact snapshot fact without filesystem I/O.
 *
 * Absent input or any identity difference returns false.
 */
int yvex_artifact_snapshot_equal(const yvex_artifact_snapshot *left,
                                 const yvex_artifact_snapshot *right)
{
    return left && right && left->device == right->device &&
           left->inode == right->inode && left->size == right->size &&
           left->mtime_seconds == right->mtime_seconds &&
           left->mtime_nanoseconds == right->mtime_nanoseconds &&
           left->ctime_seconds == right->ctime_seconds &&
           left->ctime_nanoseconds == right->ctime_nanoseconds;
}
/*
 * Starts one empty exact-file identity stream without allocation or IO.
 *
 * Physical identity does not prove semantic completeness or support.
 */
void yvex_artifact_identity_stream_init(yvex_artifact_identity_stream *stream) {
    if (!stream)
        return;
    memset(stream, 0, sizeof(*stream));
    yvex_sha256_init(&stream->hash);
    stream->active = 1;
}
/*
 * Appends one ordered byte range and checks aggregate length arithmetic.
 *
 * Physical identity does not prove semantic completeness or support.
 */
int yvex_artifact_identity_stream_update(yvex_artifact_identity_stream *stream,
                                         const unsigned char *bytes,
                                         size_t byte_count,
                                         yvex_error *err) {
    if (!stream || !stream->active || (!bytes && byte_count)) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "artifact_identity.stream.update",
                       "active stream and byte range are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (stream->bytes > ULLONG_MAX - (unsigned long long)byte_count ||
        !yvex_sha256_update(&stream->hash, bytes, byte_count)) {
        stream->active = 0;
        yvex_error_set(err,
                       YVEX_ERR_BOUNDS,
                       "artifact_identity.stream.update",
                       "artifact identity byte count overflowed");
        return YVEX_ERR_BOUNDS;
    }
    stream->bytes += (unsigned long long)byte_count;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Finalizes only exact expected coverage and clears mutable hash state.
 *
 * Physical identity does not prove semantic completeness or support.
 */
int yvex_artifact_identity_stream_final(yvex_artifact_identity_stream *stream,
                                        unsigned long long expected_bytes,
                                        char out_hex[YVEX_SHA256_HEX_CAP],
                                        yvex_error *err) {
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (out_hex)
        out_hex[0] = '\0';
    if (!stream || !stream->active || !out_hex || stream->bytes != expected_bytes ||
        !yvex_sha256_final(&stream->hash, digest)) {
        if (stream)
            stream->active = 0;
        yvex_error_set(err,
                       YVEX_ERR_BOUNDS,
                       "artifact_identity.stream.final",
                       "artifact identity requires exact complete byte coverage");
        return YVEX_ERR_BOUNDS;
    }
    yvex_sha256_hex(digest, out_hex);
    memset(stream, 0, sizeof(*stream));
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Project sha256 hex bytes facts while preserving the canonical artifact identity invariants.
 *
 * Physical identity does not prove semantic completeness or support.
 */
int yvex_artifact_sha256_hex_bytes(const unsigned char *data,
                                   unsigned long long len,
                                   char out_hex[YVEX_SHA256_HEX_CAP],
                                   yvex_error *err) {
    artifact_hash_provider provider;
    artifact_hash hash;
    unsigned char digest[32];
    int ok;
    if (!data || !out_hex) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "yvex_artifact_sha256_hex_bytes",
                       "data and out_hex are required");
        return YVEX_ERR_INVALID_ARG;
    }
    artifact_hash_provider_open(&provider);
    artifact_hash_init(&hash, &provider);
    ok = len <= (unsigned long long)SIZE_MAX &&
         artifact_hash_update(&hash, data, (size_t)len) &&
         artifact_hash_final(&hash, digest);
    artifact_hash_abort(&hash);
    artifact_hash_provider_close(&provider);
    if (!ok) {
        yvex_error_set(err,
                       YVEX_ERR_BOUNDS,
                       "yvex_artifact_sha256_hex_bytes",
                       "input length exceeds SHA-256 addressable bounds");
        return YVEX_ERR_BOUNDS;
    }
    yvex_sha256_hex(digest, out_hex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Hash the complete artifact byte sequence while detecting short reads and replacement.
 *
 * Physical identity does not prove semantic completeness or support.
 */
int yvex_artifact_compute_sha256(const char *path,
                                 char out_hex[YVEX_SHA256_HEX_CAP],
                                 yvex_error *err) {
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    int rc;
    if (!path || !path[0] || !out_hex) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_artifact_compute_sha256",
                       "path and out_hex are required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = artifact_hash_path(path, 0u, digest, NULL, err);
    if (rc != YVEX_OK)
        return rc;
    yvex_sha256_hex(digest, out_hex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Hash one exact artifact span through a caller-supplied bounded reader. Physical identity does
 * not prove semantic completeness or support.
 */
int yvex_artifact_identity_read(const char *path,
                                yvex_artifact_file_identity *out,
                                yvex_error *err) {
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    int n;
    int rc;
    if (!path || !path[0] || !out) {
        yvex_error_set(
            err, YVEX_ERR_INVALID_ARG, "yvex_artifact_identity_read", "path and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    n = snprintf(out->path, sizeof(out->path), "%s", path);
    if (n < 0 || (unsigned int)n >= sizeof(out->path)) {
        yvex_error_set(
            err, YVEX_ERR_BOUNDS, "yvex_artifact_identity_read", "artifact path too long");
        return YVEX_ERR_BOUNDS;
    }
    rc = artifact_hash_path(path, 1u, digest, &out->file_size, err);
    if (rc != YVEX_OK)
        return rc;
    yvex_sha256_hex(digest, out->sha256);
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Hash the exact borrowed handle between pre-read and post-read identity checks. Physical identity
 * does not prove semantic completeness or support.
 */
int yvex_artifact_identity_read_open(const yvex_artifact *artifact,
                                     yvex_artifact_file_identity *out,
                                     yvex_error *err)
{
    return yvex_artifact_identity_read_open_progress(artifact, out, NULL, NULL, err);
}
/*
 * Hash one stable opened artifact while reporting exact byte progress.
 *
 * Performs bounded reads and invokes the callback after each completed chunk. Callback
 * cancellation or physical drift publishes no partial identity.
 */
int yvex_artifact_identity_read_open_progress(
    const yvex_artifact *artifact, yvex_artifact_file_identity *out,
    int (*progress)(void *context, unsigned long long completed,
                    unsigned long long total),
    void *progress_context, yvex_error *err)
{
    artifact_hash_provider provider;
    artifact_hash hash;
    unsigned char digest[32];
    unsigned char buf[65536];
    unsigned long long size;
    int n;
    int rc;
    if (!artifact || !out) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "yvex_artifact_identity_read_open",
                       "artifact and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK)
        return rc;
    n = snprintf(out->path, sizeof(out->path), "%s", yvex_artifact_path(artifact));
    if (n < 0 || (unsigned int)n >= sizeof(out->path)) {
        yvex_error_set(
            err, YVEX_ERR_BOUNDS, "yvex_artifact_identity_read_open", "artifact path too long");
        return YVEX_ERR_BOUNDS;
    }
    size = yvex_artifact_size(artifact);
    artifact_hash_provider_open(&provider);
    artifact_hash_init(&hash, &provider);
    rc = artifact_hash_range(artifact, 0u, size, buf, sizeof(buf), &hash,
                             progress, progress_context, err);
    if (rc != YVEX_OK) {
        if (!yvex_error_is_set(err))
            yvex_error_set(err, rc, "yvex_artifact_identity_read_open",
                           "artifact SHA-256 length overflow");
        goto failure;
    }
    rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK)
        goto failure;
    if (!artifact_hash_final(&hash, digest)) {
        yvex_error_set(err,
                       YVEX_ERR_BOUNDS,
                       "yvex_artifact_identity_read_open",
                       "artifact SHA-256 finalization failed");
        rc = YVEX_ERR_BOUNDS;
        goto failure;
    }
    rc = yvex_artifact_cache_release(artifact, 0ull, size, err);
    if (rc != YVEX_OK)
        goto failure;
    artifact_hash_provider_close(&provider);
    yvex_sha256_hex(digest, out->sha256);
    out->file_size = size;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    artifact_hash_abort(&hash);
    artifact_hash_provider_close(&provider);
    memset(out, 0, sizeof(*out));
    return rc;
}
/*
 * Hash ordered raw tensor payloads independently from semantic plan identities.
 *
 * Stable opened artifact, parsed GGUF directory, and bounded streaming budget. Malformed ranges,
 * short reads, allocation, or snapshot drift publish no identity. Excludes metadata and padding;
 * it does not establish semantic compatibility alone.
 */
int yvex_artifact_payload_identity_compute(const yvex_artifact *artifact, const yvex_gguf *gguf,
                                           size_t buffer_bytes,
                                           yvex_artifact_payload_identity *out, yvex_error *err)
{
    yvex_artifact_payload_identity next = {0};
    artifact_hash_provider provider;
    artifact_hash terminal = {0};
    yvex_sha256 aggregate;
    unsigned char aggregate_digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned char *buffer = NULL;
    yvex_gguf_layout_result layout;
    unsigned long long ordinal;
    int rc = YVEX_OK;
    if (out) memset(out, 0, sizeof(*out));
    if (!artifact || !gguf || !out || !buffer_bytes || buffer_bytes > (size_t)SSIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "artifact.payload.identity",
                       "artifact, GGUF directory, buffer budget, and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    next.tensor_count = yvex_gguf_tensor_count(gguf);
    if (!next.tensor_count) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "artifact.payload.identity",
                       "non-empty GGUF tensor payload is required");
        return YVEX_ERR_FORMAT;
    }
    memset(&layout, 0, sizeof(layout));
    rc = yvex_gguf_layout_validate(artifact, gguf, &layout, err);
    if (rc != YVEX_OK || !layout.accepted || layout.tensor_payload_bytes_read != 0ull) {
        if (rc == YVEX_OK) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "artifact.payload.identity",
                           "canonical non-overlapping GGUF layout is required");
            rc = YVEX_ERR_FORMAT;
        }
        return rc;
    }
    rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK)
        return rc;
    buffer = malloc(buffer_bytes);
    if (!buffer) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "artifact.payload.identity",
                       "payload identity streaming buffer allocation failed");
        return YVEX_ERR_NOMEM;
    }
    artifact_hash_provider_open(&provider);
    yvex_sha256_init(&aggregate);
    if (!yvex_sha256_update_text(&aggregate, "yvex.quant.payload.bytes.v1") ||
        !yvex_sha256_update_u64(&aggregate, next.tensor_count)) {
        rc = YVEX_ERR_BOUNDS;
        goto done;
    }
    for (ordinal = 0ull; ordinal < next.tensor_count; ++ordinal) {
        const yvex_gguf_tensor_info *tensor = yvex_gguf_tensor_at(gguf, ordinal);
        unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
        char terminal_hex[YVEX_SHA256_HEX_CAP];
        if (!tensor || !tensor->range_addressable || !tensor->storage_bytes) {
            rc = YVEX_ERR_FORMAT;
            goto done;
        }
        artifact_hash_init(&terminal, &provider);
        rc = artifact_hash_range(artifact, tensor->absolute_offset, tensor->storage_bytes, buffer,
                                 buffer_bytes, &terminal, NULL, NULL, err);
        if (rc != YVEX_OK)
            goto done;
        if (ULLONG_MAX - next.payload_bytes_read < tensor->storage_bytes ||
            !artifact_hash_final(&terminal, digest)) {
            rc = YVEX_ERR_BOUNDS;
            goto done;
        }
        next.payload_bytes_read += tensor->storage_bytes;
        yvex_sha256_hex(digest, terminal_hex);
        if (!yvex_sha256_update_u64(&aggregate, ordinal) ||
            !yvex_sha256_update_u64(&aggregate, tensor->ggml_type) ||
            !yvex_sha256_update_u64(&aggregate, tensor->storage_bytes) ||
            !yvex_sha256_update_text(&aggregate, terminal_hex)) {
            rc = YVEX_ERR_BOUNDS;
            goto done;
        }
    }
    rc = yvex_artifact_snapshot_validate(artifact, NULL, err);
    if (rc != YVEX_OK) {
        goto done;
    }
    if (!yvex_sha256_final(&aggregate, aggregate_digest)) {
        rc = YVEX_ERR_BOUNDS;
        goto done;
    }
    yvex_sha256_hex(aggregate_digest, next.payload_byte_identity);
    next.complete = 1;
    *out = next;
done:
    artifact_hash_abort(&terminal);
    artifact_hash_provider_close(&provider);
    free(buffer);
    if (rc != YVEX_OK && !yvex_error_is_set(err))
        yvex_error_set(err, rc, "artifact.payload.identity",
                       "tensor payload identity could not be derived exactly");
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
/*
 * Check structural validity of the supplied artifact identity facts.
 *
 * Physical identity does not prove semantic completeness or support.
 */
int yvex_sha256_hex_is_valid(const char *hex) {
    return yvex_sha256_hex_valid(hex);
}
