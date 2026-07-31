/*
 * Admit identity-bound token chunks for the production transformer API.
 *
 * Payload is canonical little-endian U32 and every file is regular, non-symlink, bounded, and
 * exact. Numeric model input only; token IDs do not establish tokenizer support.
 */
#include <yvex/internal/transformer.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/internal/core.h>
#include <yvex/internal/runtime.h>

#define TRANSFORMER_INPUT_HEADER_BYTES 512u
static const unsigned char transformer_input_magic[8] = {'Y','V','E','X','T','R','N','1'};

struct yvex_transformer_input {
    yvex_transformer_input_summary summary;
    unsigned int *tokens;
    struct stat snapshot;
    int fd, file_backed;
};

static int transformer_input_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.transformer.input", reason);
    return status;
}

static void transformer_u32_store(unsigned char *out, uint32_t value)
{
    out[0] = (unsigned char)value;
    out[1] = (unsigned char)(value >> 8);
    out[2] = (unsigned char)(value >> 16);
    out[3] = (unsigned char)(value >> 24);
}

static void transformer_u64_store(unsigned char *out, uint64_t value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index) out[index] = (unsigned char)(value >> (index * 8u));
}

static uint32_t transformer_u32_load(const unsigned char *in)
{
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static uint64_t transformer_u64_load(const unsigned char *in)
{
    uint64_t value = 0ull;
    unsigned int index;
    for (index = 0u; index < 8u; ++index) value |= (uint64_t)in[index] << (index * 8u);
    return value;
}

static int transformer_io(int fd, void *bytes, size_t count, off_t offset, int writing)
{
    size_t done = 0u;
    while (done < count) {
        ssize_t step = writing
                           ? pwrite(fd, (const unsigned char *)bytes + done, count - done,
                                    offset + (off_t)done)
                           : pread(fd, (unsigned char *)bytes + done, count - done,
                                   offset + (off_t)done);
        if (step < 0 && errno == EINTR) continue;
        if (step <= 0) return 0;
        done += (size_t)step;
    }
    return 1;
}

static int transformer_payload_digest(const unsigned int *tokens, unsigned long long count,
                                      char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES], bytes[4];
    unsigned long long index;
    if (!tokens || !count || !output) return 0;
    yvex_sha256_init(&hash);
    for (index = 0ull; index < count; ++index) {
        transformer_u32_store(bytes, tokens[index]);
        if (!yvex_sha256_update(&hash, bytes, sizeof(bytes))) return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int transformer_input_identity(const yvex_transformer_input_summary *summary,
                                      char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!summary || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.transformer.token-input.v1") ||
        !yvex_sha256_update_u64(&hash, summary->schema_version) ||
        !yvex_sha256_update_text(&hash, summary->logical_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->runtime_numeric_identity) ||
        !yvex_sha256_update_text(&hash, summary->runtime_descriptor_identity) ||
        !yvex_sha256_update_text(&hash, summary->transformer_plan_identity) ||
        !yvex_sha256_update_u64(&hash, summary->token_start) ||
        !yvex_sha256_update_u64(&hash, summary->token_count) ||
        !yvex_sha256_update_u64(&hash, summary->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, summary->payload_bytes) ||
        !yvex_sha256_update_text(&hash, summary->payload_digest) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

/*
 * Seal one input summary over exact U32 tokens.
 *
 * Typed identity/domain refusal.
 */
int yvex_transformer_input_seal(yvex_transformer_input_summary *summary,
                                const unsigned int *token_ids, yvex_error *err)
{
    unsigned long long index, bytes;
    if (!summary || !token_ids || summary->schema_version != YVEX_TRANSFORMER_INPUT_SCHEMA_V1 ||
        !summary->token_count || !summary->vocabulary_size ||
        !yvex_sha256_hex_valid(summary->logical_model_identity) ||
        !yvex_sha256_hex_valid(summary->runtime_numeric_identity) ||
        !yvex_sha256_hex_valid(summary->runtime_descriptor_identity) ||
        !yvex_sha256_hex_valid(summary->transformer_plan_identity) ||
        !yvex_core_u64_mul(summary->token_count, sizeof(uint32_t), &bytes))
        return transformer_input_refuse(err, YVEX_ERR_INVALID_ARG,
                                        "transformer token input facts are invalid");
    for (index = 0ull; index < summary->token_count; ++index)
        if ((unsigned long long)token_ids[index] >= summary->vocabulary_size)
            return transformer_input_refuse(err, YVEX_ERR_BOUNDS,
                                            "transformer token ID exceeds the admitted vocabulary");
    summary->payload_bytes = bytes;
    if (!transformer_payload_digest(token_ids, summary->token_count, summary->payload_digest) ||
        !transformer_input_identity(summary, summary->input_identity))
        return transformer_input_refuse(err, YVEX_ERR_STATE,
                                        "transformer token input identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int transformer_header_encode(unsigned char header[TRANSFORMER_INPUT_HEADER_BYTES],
                                     const yvex_transformer_input_summary *summary)
{
    if (!summary || !yvex_sha256_hex_valid(summary->input_identity)) return 0;
    memset(header, 0, TRANSFORMER_INPUT_HEADER_BYTES);
    memcpy(header, transformer_input_magic, sizeof(transformer_input_magic));
    transformer_u32_store(header + 8u, summary->schema_version);
    transformer_u64_store(header + 16u, summary->token_start);
    transformer_u64_store(header + 24u, summary->token_count);
    transformer_u64_store(header + 32u, summary->vocabulary_size);
    transformer_u64_store(header + 40u, summary->payload_bytes);
    memcpy(header + 48u, summary->logical_model_identity, YVEX_SHA256_HEX_CAP);
    memcpy(header + 113u, summary->runtime_numeric_identity, YVEX_SHA256_HEX_CAP);
    memcpy(header + 178u, summary->runtime_descriptor_identity, YVEX_SHA256_HEX_CAP);
    memcpy(header + 243u, summary->transformer_plan_identity, YVEX_SHA256_HEX_CAP);
    memcpy(header + 308u, summary->payload_digest, YVEX_SHA256_HEX_CAP);
    memcpy(header + 373u, summary->input_identity, YVEX_SHA256_HEX_CAP);
    return 1;
}

static int transformer_header_decode(yvex_transformer_input_summary *summary,
                                     const unsigned char header[TRANSFORMER_INPUT_HEADER_BYTES])
{
    unsigned int index;
    if (!summary || memcmp(header, transformer_input_magic, sizeof(transformer_input_magic)) != 0)
        return 0;
    for (index = 438u; index < TRANSFORMER_INPUT_HEADER_BYTES; ++index)
        if (header[index] != 0u) return 0;
    memset(summary, 0, sizeof(*summary));
    summary->schema_version = transformer_u32_load(header + 8u);
    summary->token_start = transformer_u64_load(header + 16u);
    summary->token_count = transformer_u64_load(header + 24u);
    summary->vocabulary_size = transformer_u64_load(header + 32u);
    summary->payload_bytes = transformer_u64_load(header + 40u);
    memcpy(summary->logical_model_identity, header + 48u, YVEX_SHA256_HEX_CAP);
    memcpy(summary->runtime_numeric_identity, header + 113u, YVEX_SHA256_HEX_CAP);
    memcpy(summary->runtime_descriptor_identity, header + 178u, YVEX_SHA256_HEX_CAP);
    memcpy(summary->transformer_plan_identity, header + 243u, YVEX_SHA256_HEX_CAP);
    memcpy(summary->payload_digest, header + 308u, YVEX_SHA256_HEX_CAP);
    memcpy(summary->input_identity, header + 373u, YVEX_SHA256_HEX_CAP);
    return summary->logical_model_identity[YVEX_SHA256_HEX_CAP - 1u] == '\0' &&
           summary->input_identity[YVEX_SHA256_HEX_CAP - 1u] == '\0';
}

int yvex_transformer_input_write(const char *path,
                                 const yvex_transformer_input_summary *summary,
                                 const unsigned int *token_ids, yvex_error *err)
{
    unsigned char header[TRANSFORMER_INPUT_HEADER_BYTES], bytes[4];
    yvex_transformer_input_summary sealed;
    unsigned long long index;
    int fd, rc = YVEX_OK;
    if (!path || !summary || !token_ids) return transformer_input_refuse(
        err, YVEX_ERR_INVALID_ARG, "transformer input write arguments are required");
    sealed = *summary;
    if (yvex_transformer_input_seal(&sealed, token_ids, err) != YVEX_OK ||
        strcmp(sealed.input_identity, summary->input_identity) != 0 ||
        !transformer_header_encode(header, &sealed))
        return transformer_input_refuse(err, YVEX_ERR_FORMAT,
                                        "transformer input summary is not exactly sealed");
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return transformer_input_refuse(err, YVEX_ERR_IO,
                                               "transformer input file creation failed");
    if (!transformer_io(fd, header, sizeof(header), 0, 1)) rc = YVEX_ERR_IO;
    for (index = 0ull; rc == YVEX_OK && index < sealed.token_count; ++index) {
        transformer_u32_store(bytes, token_ids[index]);
        if (!transformer_io(fd, bytes, sizeof(bytes),
                            (off_t)TRANSFORMER_INPUT_HEADER_BYTES + (off_t)(index * 4ull), 1))
            rc = YVEX_ERR_IO;
    }
    if (rc == YVEX_OK && fsync(fd) != 0) rc = YVEX_ERR_IO;
    if (close(fd) != 0 && rc == YVEX_OK) rc = YVEX_ERR_IO;
    if (rc != YVEX_OK) {
        (void)unlink(path);
        return transformer_input_refuse(err, (yvex_status)rc,
                                        "transformer input file publication failed");
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Allocate one owned memory input after exact revalidation.
 *
 * Typed identity/allocation refusal.
 */
int yvex_transformer_input_open_memory(yvex_transformer_input **out,
                                       const yvex_transformer_input_summary *summary,
                                       const unsigned int *token_ids, yvex_error *err)
{
    yvex_transformer_input *input;
    yvex_transformer_input_summary sealed;
    if (out) *out = NULL;
    if (!out || !summary || !token_ids)
        return transformer_input_refuse(err, YVEX_ERR_INVALID_ARG,
                                        "transformer memory input arguments are required");
    sealed = *summary;
    if (yvex_transformer_input_seal(&sealed, token_ids, err) != YVEX_OK ||
        strcmp(sealed.input_identity, summary->input_identity) != 0)
        return transformer_input_refuse(err, YVEX_ERR_FORMAT,
                                        "transformer memory input identity is stale");
    input = (yvex_transformer_input *)calloc(1u, sizeof(*input));
    if (!input) return transformer_input_refuse(err, YVEX_ERR_NOMEM,
                                                "transformer memory input allocation failed");
    input->fd = -1;
    input->tokens = (unsigned int *)malloc((size_t)sealed.payload_bytes);
    if (!input->tokens) {
        free(input);
        return transformer_input_refuse(err, YVEX_ERR_NOMEM,
                                        "transformer token payload allocation failed");
    }
    memcpy(input->tokens, token_ids, (size_t)sealed.payload_bytes);
    input->summary = sealed;
    *out = input;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_transformer_input_open_file(yvex_transformer_input **out, const char *path,
                                     const yvex_transformer_input_limits *limits,
                                     yvex_error *err)
{
    yvex_transformer_input *input = NULL;
    unsigned char header[TRANSFORMER_INPUT_HEADER_BYTES], bytes[4];
    unsigned long long file_bytes, index;
    struct stat link_status;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!out || !path || lstat(path, &link_status) != 0 || S_ISLNK(link_status.st_mode))
        return transformer_input_refuse(err, YVEX_ERR_INVALID_ARG,
                                        "transformer input path must be a non-symlink file");
    input = (yvex_transformer_input *)calloc(1u, sizeof(*input));
    if (!input) return transformer_input_refuse(err, YVEX_ERR_NOMEM,
                                                "transformer file input allocation failed");
    input->fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (input->fd < 0 || fstat(input->fd, &input->snapshot) != 0 ||
        !S_ISREG(input->snapshot.st_mode) || input->snapshot.st_size < 0) {
        rc = transformer_input_refuse(err, YVEX_ERR_IO,
                                      "transformer input file open or type check failed");
        goto failure;
    }
    file_bytes = (unsigned long long)input->snapshot.st_size;
    if ((limits && limits->maximum_file_bytes && file_bytes > limits->maximum_file_bytes) ||
        file_bytes < TRANSFORMER_INPUT_HEADER_BYTES ||
        !transformer_io(input->fd, header, sizeof(header), 0, 0) ||
        !transformer_header_decode(&input->summary, header) ||
        !input->summary.token_count ||
        input->summary.token_count > ULLONG_MAX / sizeof(uint32_t) ||
        input->summary.payload_bytes != input->summary.token_count * sizeof(uint32_t) ||
        input->summary.payload_bytes > SIZE_MAX ||
        file_bytes != TRANSFORMER_INPUT_HEADER_BYTES + input->summary.payload_bytes) {
        rc = transformer_input_refuse(err, YVEX_ERR_FORMAT,
                                      "transformer input file extent or header is invalid");
        goto failure;
    }
    input->tokens = (unsigned int *)malloc((size_t)input->summary.payload_bytes);
    if (!input->tokens) {
        rc = transformer_input_refuse(err, YVEX_ERR_NOMEM,
                                      "transformer file token payload allocation failed");
        goto failure;
    }
    for (index = 0ull; index < input->summary.token_count; ++index) {
        if (!transformer_io(input->fd, bytes, sizeof(bytes),
                            (off_t)TRANSFORMER_INPUT_HEADER_BYTES + (off_t)(index * 4ull), 0)) {
            rc = transformer_input_refuse(err, YVEX_ERR_IO,
                                          "transformer token payload read failed");
            goto failure;
        }
        input->tokens[index] = transformer_u32_load(bytes);
    }
    input->file_backed = 1;
    *out = input;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    yvex_transformer_input_close(&input);
    return rc;
}

int yvex_transformer_input_validate(const yvex_transformer_input *input,
                                    const yvex_transformer_plan *plan,
                                    const yvex_runtime_binding_summary *binding,
                                    yvex_error *err)
{
    yvex_transformer_input_summary sealed;
    const yvex_transformer_plan_summary *plan_summary =
        yvex_transformer_plan_summary_get(plan);
    struct stat current;
    if (!input || !plan_summary || !binding)
        return transformer_input_refuse(err, YVEX_ERR_INVALID_ARG,
                                        "transformer input validation owners are required");
    sealed = input->summary;
    if (yvex_transformer_input_seal(&sealed, input->tokens, err) != YVEX_OK ||
        strcmp(sealed.input_identity, input->summary.input_identity) != 0 ||
        strcmp(sealed.logical_model_identity, binding->logical_model_identity) != 0 ||
        strcmp(sealed.runtime_numeric_identity, binding->runtime_numeric_identity) != 0 ||
        strcmp(sealed.runtime_descriptor_identity, binding->runtime_descriptor_identity) != 0 ||
        strcmp(sealed.transformer_plan_identity,
               plan_summary->transformer_plan_identity) != 0 ||
        sealed.vocabulary_size != plan_summary->vocabulary_size)
        return transformer_input_refuse(err, YVEX_ERR_STATE,
                                        "transformer token input identity is incompatible");
    if (input->file_backed &&
        (input->fd < 0 || fstat(input->fd, &current) != 0 ||
         current.st_dev != input->snapshot.st_dev || current.st_ino != input->snapshot.st_ino ||
         current.st_size != input->snapshot.st_size ||
         current.st_mtim.tv_sec != input->snapshot.st_mtim.tv_sec ||
         current.st_mtim.tv_nsec != input->snapshot.st_mtim.tv_nsec ||
         current.st_ctim.tv_sec != input->snapshot.st_ctim.tv_sec ||
         current.st_ctim.tv_nsec != input->snapshot.st_ctim.tv_nsec))
        return transformer_input_refuse(err, YVEX_ERR_STATE,
                                        "transformer token input file drifted after admission");
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Borrow one immutable input summary.
 *
 * Borrowed lifetime ends on input close.
 */
const yvex_transformer_input_summary *yvex_transformer_input_summary_get(
    const yvex_transformer_input *input)
{
    return input ? &input->summary : NULL;
}

/*
 * Borrow canonical host-order token IDs.
 *
 * Borrowed lifetime ends on input close.
 */
const unsigned int *yvex_transformer_input_token_ids(const yvex_transformer_input *input)
{
    return input ? input->tokens : NULL;
}

void yvex_transformer_input_close(yvex_transformer_input **input)
{
    if (!input || !*input) return;
    if ((*input)->fd >= 0) (void)close((*input)->fd);
    free((*input)->tokens);
    memset(*input, 0, sizeof(**input));
    free(*input);
    *input = NULL;
}
