/* Encode the semantic session prefix separately from runtime-owned model-state pages. */
#include "src/server/private.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

#define SESSION_STORE_MAGIC "YVSESS01"
#define SESSION_STORE_MAGIC_BYTES 8u
#define SESSION_STORE_FIXED_BYTES 864ull
#define SESSION_STORE_DIGEST_BYTES YVEX_SHA256_DIGEST_BYTES

typedef struct {
    unsigned char *bytes;
    size_t capacity, offset;
} session_store_writer;

typedef struct {
    const unsigned char *bytes;
    size_t count, offset, digest_offset;
} session_store_parser;

static int session_store_refuse(yvex_error *err, yvex_status status,
                                const char *reason)
{
    yvex_error_set(err, status, "server.session-store", reason);
    return status;
}

static int session_store_add(unsigned long long left,
                             unsigned long long right,
                             unsigned long long *result)
{
    return yvex_core_u64_add(left, right, result);
}

static int session_store_align(unsigned long long value,
                               unsigned long long *result)
{
    unsigned long long adjusted;
    if (!yvex_core_u64_add(value, 7ull, &adjusted)) return 0;
    *result = adjusted & ~7ull;
    return 1;
}

static int session_store_optional_identity_valid(const char *identity)
{
    return identity && (!identity[0] || yvex_sha256_hex_valid(identity));
}

static uint64_t session_store_double_bits(double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double session_store_bits_double(uint64_t bits)
{
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int session_store_writer_take(session_store_writer *writer,
                                     size_t count, unsigned char **out)
{
    if (!writer || !out || count > writer->capacity - writer->offset) return 0;
    *out = writer->bytes + writer->offset;
    writer->offset += count;
    return 1;
}

static int session_store_writer_bytes(session_store_writer *writer,
                                      const void *bytes, size_t count)
{
    unsigned char *destination;
    if ((!bytes && count) || !session_store_writer_take(writer, count, &destination))
        return 0;
    if (count) memcpy(destination, bytes, count);
    return 1;
}

static int session_store_writer_u64(session_store_writer *writer,
                                    unsigned long long value)
{
    return session_store_writer_bytes(writer, &value, sizeof(value));
}

static int session_store_writer_identity(session_store_writer *writer,
                                         const char *identity)
{
    static const unsigned char absent[64] = {0};
    if (!session_store_optional_identity_valid(identity)) return 0;
    return identity[0]
               ? session_store_writer_bytes(writer, identity, 64u)
               : session_store_writer_bytes(writer, absent, sizeof(absent));
}

static int session_store_writer_blob(session_store_writer *writer,
                                     const void *bytes,
                                     unsigned long long byte_count)
{
    unsigned long long aligned;
    unsigned char *destination;
    if (byte_count > SIZE_MAX || (!bytes && byte_count) ||
        !session_store_align(byte_count, &aligned) || aligned > SIZE_MAX ||
        !session_store_writer_take(writer, (size_t)aligned, &destination))
        return 0;
    if (byte_count) memcpy(destination, bytes, (size_t)byte_count);
    if (aligned > byte_count)
        memset(destination + byte_count, 0, (size_t)(aligned - byte_count));
    return 1;
}

static int session_store_parser_take(session_store_parser *parser, size_t count,
                                     const unsigned char **out)
{
    if (!parser || !out || parser->offset > parser->digest_offset ||
        count > parser->digest_offset - parser->offset)
        return 0;
    *out = parser->bytes + parser->offset;
    parser->offset += count;
    return 1;
}

static int session_store_parser_u64(session_store_parser *parser,
                                    unsigned long long *value)
{
    const unsigned char *bytes;
    if (!value || !session_store_parser_take(parser, sizeof(*value), &bytes))
        return 0;
    memcpy(value, bytes, sizeof(*value));
    return 1;
}

static int session_store_parser_identity(
    session_store_parser *parser, char identity[YVEX_SHA256_HEX_CAP])
{
    const unsigned char *bytes;
    size_t index;
    if (!identity || !session_store_parser_take(parser, 64u, &bytes)) return 0;
    for (index = 0u; index < 64u && bytes[index] == 0u; ++index) {}
    if (index == 64u) {
        identity[0] = '\0';
        return 1;
    }
    memcpy(identity, bytes, 64u);
    identity[64] = '\0';
    return yvex_sha256_hex_valid(identity);
}

static int session_store_parser_blob(session_store_parser *parser,
                                     unsigned long long byte_count,
                                     const unsigned char **bytes)
{
    const unsigned char *payload;
    unsigned long long aligned;
    size_t index;
    if (!bytes || byte_count > SIZE_MAX || !session_store_align(byte_count, &aligned) ||
        aligned > SIZE_MAX ||
        !session_store_parser_take(parser, (size_t)aligned, &payload))
        return 0;
    for (index = (size_t)byte_count; index < (size_t)aligned; ++index)
        if (payload[index] != 0u) return 0;
    *bytes = payload;
    return 1;
}

static int session_store_checkpoint_empty(
    const yvex_runtime_generation_checkpoint *checkpoint)
{
    const unsigned char *bytes = (const unsigned char *)checkpoint;
    size_t index;
    for (index = 0u; index < sizeof(*checkpoint); ++index)
        if (bytes[index] != 0u) return 0;
    return 1;
}

static int session_store_view_valid(const server_session_store_view *view)
{
    unsigned long long index;
    if (!view || view->policy_set < 0 || view->policy_set > 1 ||
        view->generation_checkpoint_present < 0 ||
        view->generation_checkpoint_present > 1 ||
        !yvex_reasoning_policy_valid(view->reasoning_policy) ||
        (view->message_count && !view->messages) ||
        (view->committed_count && !view->committed_tokens) ||
        (view->committed_count &&
         (!view->policy_set || !view->generation_checkpoint_present)) ||
        !session_store_optional_identity_valid(view->last_turn_identity) ||
        !session_store_optional_identity_valid(view->state_digest) ||
        !session_store_optional_identity_valid(view->generated_token_identity) ||
        !session_store_optional_identity_valid(view->generated_text_digest))
        return 0;
    if (view->policy_set) {
        if (!yvex_sha256_hex_valid(view->policy.policy_identity)) return 0;
    } else if (memcmp(&view->policy, &(yvex_runtime_sampling_policy){0},
                      sizeof(view->policy)) != 0) {
        return 0;
    }
    if (view->generation_checkpoint_present) {
        if (view->generation_checkpoint.schema_version !=
                YVEX_RUNTIME_GENERATION_CHECKPOINT_SCHEMA_V1 ||
            view->generation_checkpoint.sampling.schema_version !=
                YVEX_RUNTIME_SAMPLING_CHECKPOINT_SCHEMA_V1 ||
            !yvex_sha256_hex_valid(
                view->generation_checkpoint.sampling.policy_identity) ||
            !yvex_sha256_hex_valid(
                view->generation_checkpoint.sampling.rng_state_identity) ||
            !yvex_sha256_hex_valid(
                view->generation_checkpoint.sampling.checkpoint_identity) ||
            !yvex_sha256_hex_valid(
                view->generation_checkpoint.generation_plan_identity) ||
            !yvex_sha256_hex_valid(
                view->generation_checkpoint.checkpoint_identity) ||
            !view->policy_set ||
            strcmp(view->generation_checkpoint.sampling.policy_identity,
                   view->policy.policy_identity) != 0)
            return 0;
    } else if (!session_store_checkpoint_empty(&view->generation_checkpoint)) {
        return 0;
    }
    for (index = 0ull; index < view->message_count; ++index)
        if (view->messages[index].schema_version !=
                YVEX_PROMPT_MESSAGE_SCHEMA_V1 ||
            view->messages[index].role > YVEX_PROMPT_ROLE_TOOL ||
            (view->messages[index].content_len && !view->messages[index].content) ||
            (view->messages[index].reasoning_content_len &&
             !view->messages[index].reasoning_content))
            return 0;
    return 1;
}

static int session_store_measure(const server_session_store_view *view,
                                 unsigned long long *total)
{
    unsigned long long bytes = SESSION_STORE_FIXED_BYTES;
    unsigned long long index, aligned, tokens;
    for (index = 0ull; index < view->message_count; ++index) {
        unsigned long long reasoning_aligned;
        if (!session_store_align(view->messages[index].content_len, &aligned) ||
            !session_store_align(
                view->messages[index].reasoning_content_len,
                &reasoning_aligned) ||
            !session_store_add(bytes, 24ull, &bytes) ||
            !session_store_add(bytes, reasoning_aligned, &bytes) ||
            !session_store_add(bytes, aligned, &bytes))
            return 0;
    }
    if (!yvex_core_u64_mul(view->committed_count, 8ull, &tokens) ||
        !session_store_add(bytes, tokens, &bytes) ||
        !session_store_add(bytes, SESSION_STORE_DIGEST_BYTES, &bytes) ||
        bytes > SIZE_MAX)
        return 0;
    *total = bytes;
    return 1;
}

static int session_store_policy_write(session_store_writer *writer,
                                      const yvex_runtime_sampling_policy *policy)
{
    return session_store_writer_u64(writer, policy->schema_version) &&
           session_store_writer_u64(writer, policy->strategy) &&
           session_store_writer_u64(writer,
                                    session_store_double_bits(policy->temperature)) &&
           session_store_writer_u64(writer, policy->top_k) &&
           session_store_writer_u64(writer, session_store_double_bits(policy->top_p)) &&
           session_store_writer_u64(writer, session_store_double_bits(policy->min_p)) &&
           session_store_writer_u64(writer,
                                    session_store_double_bits(policy->typical_p)) &&
           session_store_writer_u64(writer, (unsigned long long)policy->seed_present) &&
           session_store_writer_u64(writer, policy->seed) &&
           session_store_writer_u64(writer, policy->rng_algorithm) &&
           session_store_writer_u64(writer, policy->rng_version) &&
           session_store_writer_u64(writer, policy->filter_order_version) &&
           session_store_writer_identity(writer, policy->policy_identity);
}

static int session_store_checkpoint_write(
    session_store_writer *writer,
    const yvex_runtime_generation_checkpoint *checkpoint)
{
    return session_store_writer_u64(writer, checkpoint->schema_version) &&
           session_store_writer_u64(writer, checkpoint->sampling.schema_version) &&
           session_store_writer_u64(writer, checkpoint->sampling.rng_state) &&
           session_store_writer_u64(writer, checkpoint->sampling.rng_increment) &&
           session_store_writer_u64(writer, checkpoint->sampling.successful_draws) &&
           session_store_writer_identity(
               writer, checkpoint->sampling.policy_identity) &&
           session_store_writer_identity(
               writer, checkpoint->sampling.rng_state_identity) &&
           session_store_writer_identity(
               writer, checkpoint->sampling.checkpoint_identity) &&
           session_store_writer_identity(
               writer, checkpoint->generation_plan_identity) &&
           session_store_writer_identity(writer, checkpoint->checkpoint_identity);
}

int yvex_server_session_store_encode(
    const server_session_store_view *view, unsigned char **bytes,
    unsigned long long *byte_count,
    char payload_identity[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_runtime_generation_checkpoint absent = {0};
    yvex_runtime_sampling_policy no_policy = {0};
    session_store_writer writer;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long total, index;
    if (bytes) *bytes = NULL;
    if (byte_count) *byte_count = 0ull;
    if (payload_identity) payload_identity[0] = '\0';
    if (!bytes || !byte_count || !payload_identity ||
        !session_store_view_valid(view) || !session_store_measure(view, &total))
        return session_store_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "session checkpoint view is invalid");
    memset(&writer, 0, sizeof(writer));
    writer.bytes = yvex_core_calloc(1u, (size_t)total);
    writer.capacity = (size_t)total;
    if (!writer.bytes)
        return session_store_refuse(err, YVEX_ERR_NOMEM,
                                    "session checkpoint allocation failed");
    if (!session_store_writer_bytes(&writer, SESSION_STORE_MAGIC,
                                    SESSION_STORE_MAGIC_BYTES) ||
        !session_store_writer_u64(&writer, YVEX_SERVER_SESSION_STORE_SCHEMA_V2) ||
        !session_store_writer_u64(&writer, total) ||
        !session_store_writer_u64(&writer, view->message_count) ||
        !session_store_writer_u64(&writer, view->committed_count) ||
        !session_store_writer_u64(&writer, view->turn_count) ||
        !session_store_writer_u64(&writer, view->message_history_generation) ||
        !session_store_writer_u64(&writer, view->transcript_generation) ||
        !session_store_writer_u64(&writer, (unsigned long long)view->policy_set) ||
        !session_store_writer_u64(&writer, view->reasoning_policy) ||
        !session_store_writer_u64(
            &writer, (unsigned long long)view->generation_checkpoint_present) ||
        !session_store_writer_identity(&writer, view->last_turn_identity) ||
        !session_store_writer_identity(&writer, view->state_digest) ||
        !session_store_writer_identity(&writer, view->generated_token_identity) ||
        !session_store_writer_identity(&writer, view->generated_text_digest) ||
        !session_store_policy_write(&writer,
                                    view->policy_set ? &view->policy : &no_policy) ||
        !session_store_checkpoint_write(
            &writer, view->generation_checkpoint_present
                         ? &view->generation_checkpoint
                         : &absent))
        goto failure;
    for (index = 0ull; index < view->message_count; ++index)
        if (!session_store_writer_u64(&writer, view->messages[index].role) ||
            !session_store_writer_u64(&writer,
                                      view->messages[index].content_len) ||
            !session_store_writer_u64(
                &writer, view->messages[index].reasoning_content_len) ||
            !session_store_writer_blob(&writer, view->messages[index].content,
                                       view->messages[index].content_len) ||
            !session_store_writer_blob(
                &writer, view->messages[index].reasoning_content,
                view->messages[index].reasoning_content_len))
            goto failure;
    for (index = 0ull; index < view->committed_count; ++index)
        if (!session_store_writer_u64(&writer, view->committed_tokens[index]))
            goto failure;
    if (writer.offset != writer.capacity - SESSION_STORE_DIGEST_BYTES)
        goto failure;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, writer.bytes, writer.offset) ||
        !yvex_sha256_final(&hash, digest) ||
        !session_store_writer_bytes(&writer, digest, sizeof(digest)) ||
        writer.offset != writer.capacity)
        goto failure;
    yvex_sha256_hex(digest, payload_identity);
    *bytes = writer.bytes;
    *byte_count = total;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    yvex_core_free(writer.bytes);
    return session_store_refuse(err, YVEX_ERR_STATE,
                                "session checkpoint encoding diverged");
}

static int session_store_policy_read(session_store_parser *parser,
                                     yvex_runtime_sampling_policy *policy)
{
    unsigned long long fields[12];
    unsigned int index;
    memset(policy, 0, sizeof(*policy));
    for (index = 0u; index < 12u; ++index)
        if (!session_store_parser_u64(parser, &fields[index])) return 0;
    if (fields[0] > UINT_MAX || fields[1] > UINT_MAX || fields[7] > 1ull ||
        fields[9] > UINT_MAX || fields[10] > UINT_MAX || fields[11] > UINT_MAX)
        return 0;
    policy->schema_version = (unsigned int)fields[0];
    policy->strategy = (yvex_sampling_strategy)fields[1];
    policy->temperature = session_store_bits_double(fields[2]);
    policy->top_k = fields[3];
    policy->top_p = session_store_bits_double(fields[4]);
    policy->min_p = session_store_bits_double(fields[5]);
    policy->typical_p = session_store_bits_double(fields[6]);
    policy->seed_present = (int)fields[7];
    policy->seed = fields[8];
    policy->rng_algorithm = (unsigned int)fields[9];
    policy->rng_version = (unsigned int)fields[10];
    policy->filter_order_version = (unsigned int)fields[11];
    return session_store_parser_identity(parser, policy->policy_identity);
}

static int session_store_checkpoint_read(
    session_store_parser *parser,
    yvex_runtime_generation_checkpoint *checkpoint)
{
    unsigned long long fields[5];
    unsigned int index;
    memset(checkpoint, 0, sizeof(*checkpoint));
    for (index = 0u; index < 5u; ++index)
        if (!session_store_parser_u64(parser, &fields[index])) return 0;
    if (fields[0] > UINT_MAX || fields[1] > UINT_MAX) return 0;
    checkpoint->schema_version = (unsigned int)fields[0];
    checkpoint->sampling.schema_version = (unsigned int)fields[1];
    checkpoint->sampling.rng_state = fields[2];
    checkpoint->sampling.rng_increment = fields[3];
    checkpoint->sampling.successful_draws = fields[4];
    return session_store_parser_identity(
               parser, checkpoint->sampling.policy_identity) &&
           session_store_parser_identity(
               parser, checkpoint->sampling.rng_state_identity) &&
           session_store_parser_identity(
               parser, checkpoint->sampling.checkpoint_identity) &&
           session_store_parser_identity(
               parser, checkpoint->generation_plan_identity) &&
           session_store_parser_identity(parser, checkpoint->checkpoint_identity);
}

static int session_store_digest_valid(const unsigned char *bytes, size_t count,
                                      unsigned char digest[YVEX_SHA256_DIGEST_BYTES])
{
    yvex_sha256 hash;
    if (count < SESSION_STORE_DIGEST_BYTES) return 0;
    yvex_sha256_init(&hash);
    return yvex_sha256_update(&hash, bytes, count - SESSION_STORE_DIGEST_BYTES) &&
           yvex_sha256_final(&hash, digest) &&
           memcmp(digest, bytes + count - SESSION_STORE_DIGEST_BYTES,
                  SESSION_STORE_DIGEST_BYTES) == 0;
}

static int session_store_header_read(
    session_store_parser *parser, server_session_store_state *state,
    unsigned long long expected_bytes)
{
    const unsigned char *magic;
    unsigned long long schema, total, fields[8];
    unsigned int index;
    if (!session_store_parser_take(parser, SESSION_STORE_MAGIC_BYTES, &magic) ||
        memcmp(magic, SESSION_STORE_MAGIC, SESSION_STORE_MAGIC_BYTES) != 0 ||
        !session_store_parser_u64(parser, &schema) ||
        !session_store_parser_u64(parser, &total))
        return 0;
    for (index = 0u; index < 8u; ++index)
        if (!session_store_parser_u64(parser, &fields[index])) return 0;
    if ((schema != YVEX_SERVER_SESSION_STORE_SCHEMA_V1 &&
         schema != YVEX_SERVER_SESSION_STORE_SCHEMA_V2) ||
        total != expected_bytes ||
        fields[5] > 1ull ||
        !yvex_reasoning_policy_valid((yvex_reasoning_policy)fields[6]) ||
        fields[7] > 1ull)
        return 0;
    state->schema_version = schema;
    state->message_count = fields[0];
    state->committed_count = fields[1];
    state->turn_count = fields[2];
    state->message_history_generation = fields[3];
    state->transcript_generation = fields[4];
    state->policy_set = (int)fields[5];
    state->reasoning_policy = (yvex_reasoning_policy)fields[6];
    state->generation_checkpoint_present = (int)fields[7];
    return session_store_parser_identity(parser, state->last_turn_identity) &&
           session_store_parser_identity(parser, state->state_digest) &&
           session_store_parser_identity(parser, state->generated_token_identity) &&
           session_store_parser_identity(parser, state->generated_text_digest) &&
           session_store_policy_read(parser, &state->policy) &&
           session_store_checkpoint_read(parser, &state->generation_checkpoint);
}

static int session_store_decode_messages(
    session_store_parser *parser, unsigned long long maximum_transcript_bytes,
    server_session_store_state *state)
{
    session_store_parser scan = *parser;
    unsigned long long index, transcript_bytes = 0ull;
    const unsigned char *content;
    for (index = 0ull; index < state->message_count; ++index) {
        unsigned long long role, content_len, reasoning_len = 0ull, next;
        if (!session_store_parser_u64(&scan, &role) ||
            !session_store_parser_u64(&scan, &content_len) ||
            (state->schema_version >= YVEX_SERVER_SESSION_STORE_SCHEMA_V2 &&
             !session_store_parser_u64(&scan, &reasoning_len)) ||
            role > YVEX_PROMPT_ROLE_TOOL ||
            !session_store_add(content_len, 1ull, &next) ||
            !session_store_add(transcript_bytes, next, &transcript_bytes) ||
            !session_store_add(reasoning_len, 1ull, &next) ||
            !session_store_add(transcript_bytes, next, &transcript_bytes) ||
            transcript_bytes > maximum_transcript_bytes ||
            !session_store_parser_blob(&scan, content_len, &content) ||
            (state->schema_version >= YVEX_SERVER_SESSION_STORE_SCHEMA_V2 &&
             !session_store_parser_blob(&scan, reasoning_len, &content)))
            return 0;
        state->messages[index].schema_version = YVEX_PROMPT_MESSAGE_SCHEMA_V1;
        state->messages[index].role = (yvex_prompt_role)role;
        state->messages[index].content_len = content_len;
        state->messages[index].reasoning_content_len = reasoning_len;
    }
    if (transcript_bytes) {
        state->transcript = yvex_core_calloc(1u, (size_t)transcript_bytes);
        if (!state->transcript) return 0;
    }
    transcript_bytes = 0ull;
    for (index = 0ull; index < state->message_count; ++index) {
        unsigned long long role, content_len, reasoning_len = 0ull;
        if (!session_store_parser_u64(parser, &role) ||
            !session_store_parser_u64(parser, &content_len) ||
            (state->schema_version >= YVEX_SERVER_SESSION_STORE_SCHEMA_V2 &&
             !session_store_parser_u64(parser, &reasoning_len)) ||
            role != (unsigned long long)state->messages[index].role ||
            content_len != state->messages[index].content_len ||
            reasoning_len != state->messages[index].reasoning_content_len ||
            !session_store_parser_blob(parser, content_len, &content))
            return 0;
        state->messages[index].content =
            (const char *)state->transcript + transcript_bytes;
        if (content_len)
            memcpy(state->transcript + transcript_bytes, content,
                   (size_t)content_len);
        transcript_bytes += content_len + 1ull;
        state->messages[index].reasoning_content =
            (const char *)state->transcript + transcript_bytes;
        if (state->schema_version >= YVEX_SERVER_SESSION_STORE_SCHEMA_V2) {
            if (!session_store_parser_blob(parser, reasoning_len, &content))
                return 0;
            if (reasoning_len)
                memcpy(state->transcript + transcript_bytes, content,
                       (size_t)reasoning_len);
        }
        transcript_bytes += reasoning_len + 1ull;
    }
    state->transcript_count = transcript_bytes;
    return 1;
}

static int session_store_decode_tokens(session_store_parser *parser,
                                       server_session_store_state *state)
{
    unsigned long long index, token;
    if (state->committed_count) {
        if (state->committed_count > SIZE_MAX / sizeof(*state->committed_tokens))
            return 0;
        state->committed_tokens = yvex_core_calloc(
            (size_t)state->committed_count, sizeof(*state->committed_tokens));
        if (!state->committed_tokens) return 0;
    }
    for (index = 0ull; index < state->committed_count; ++index) {
        if (!session_store_parser_u64(parser, &token) || token > UINT_MAX)
            return 0;
        state->committed_tokens[index] = (unsigned int)token;
    }
    return 1;
}

int yvex_server_session_store_decode(
    const unsigned char *bytes, unsigned long long byte_count,
    unsigned long long maximum_messages,
    unsigned long long maximum_transcript_bytes,
    unsigned long long maximum_tokens, unsigned long long vocabulary_size,
    server_session_store_state *state, yvex_error *err)
{
    session_store_parser parser;
    yvex_runtime_sampling_policy canonical;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    int valid;
    if (state) memset(state, 0, sizeof(*state));
    if (!bytes || byte_count > SIZE_MAX ||
        byte_count < SESSION_STORE_FIXED_BYTES + SESSION_STORE_DIGEST_BYTES ||
        !maximum_messages || !maximum_transcript_bytes || !maximum_tokens ||
        !vocabulary_size || !state ||
        !session_store_digest_valid(bytes, (size_t)byte_count, digest))
        return session_store_refuse(err, YVEX_ERR_FORMAT,
                                    "session checkpoint is truncated or corrupt");
    memset(&parser, 0, sizeof(parser));
    parser.bytes = bytes;
    parser.count = (size_t)byte_count;
    parser.digest_offset = parser.count - SESSION_STORE_DIGEST_BYTES;
    valid = session_store_header_read(&parser, state, byte_count) &&
            state->message_count <= maximum_messages &&
            state->committed_count <= maximum_tokens &&
            (!state->committed_count ||
             (state->policy_set && state->generation_checkpoint_present));
    if (valid && state->message_count) {
        if (state->message_count > SIZE_MAX / sizeof(*state->messages))
            valid = 0;
        else
            state->messages = yvex_core_calloc(
                (size_t)state->message_count, sizeof(*state->messages));
        if (!state->messages) valid = 0;
    }
    if (valid)
        valid = session_store_decode_messages(
                    &parser, maximum_transcript_bytes, state) &&
                session_store_decode_tokens(&parser, state) &&
                parser.offset == parser.digest_offset;
    if (valid && state->policy_set) {
        canonical = state->policy;
        canonical.policy_identity[0] = '\0';
        valid = yvex_runtime_sampling_policy_seal(
                    &canonical, vocabulary_size, err) == YVEX_OK &&
                strcmp(canonical.policy_identity,
                       state->policy.policy_identity) == 0;
    } else if (valid) {
        valid = memcmp(&state->policy, &(yvex_runtime_sampling_policy){0},
                       sizeof(state->policy)) == 0;
    }
    if (valid && state->generation_checkpoint_present)
        valid = state->generation_checkpoint.schema_version ==
                    YVEX_RUNTIME_GENERATION_CHECKPOINT_SCHEMA_V1 &&
                state->generation_checkpoint.sampling.schema_version ==
                    YVEX_RUNTIME_SAMPLING_CHECKPOINT_SCHEMA_V1 &&
                yvex_sha256_hex_valid(
                    state->generation_checkpoint.sampling.policy_identity) &&
                yvex_sha256_hex_valid(
                    state->generation_checkpoint.sampling.rng_state_identity) &&
                yvex_sha256_hex_valid(
                    state->generation_checkpoint.sampling.checkpoint_identity) &&
                yvex_sha256_hex_valid(
                    state->generation_checkpoint.generation_plan_identity) &&
                yvex_sha256_hex_valid(
                    state->generation_checkpoint.checkpoint_identity) &&
                state->policy_set &&
                strcmp(state->generation_checkpoint.sampling.policy_identity,
                       state->policy.policy_identity) == 0;
    else if (valid)
        valid = session_store_checkpoint_empty(&state->generation_checkpoint);
    if (!valid) {
        yvex_server_session_store_close(state);
        return session_store_refuse(err, YVEX_ERR_FORMAT,
                                    "session checkpoint semantics are invalid");
    }
    yvex_sha256_hex(digest, state->payload_identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_server_session_store_close(server_session_store_state *state)
{
    if (!state) return;
    yvex_core_free(state->committed_tokens);
    yvex_core_free(state->transcript);
    yvex_core_free(state->messages);
    memset(state, 0, sizeof(*state));
}

void yvex_server_session_store_bytes_close(unsigned char **bytes)
{
    if (!bytes || !*bytes) return;
    yvex_core_free(*bytes);
    *bytes = NULL;
}

static void session_store_view_set(
    const server_session *session, server_session_store_view *view)
{
    memset(view, 0, sizeof(*view));
    view->messages = session->messages;
    view->committed_tokens = session->committed_tokens;
    view->message_count = session->message_count;
    view->committed_count = session->committed_count;
    view->turn_count = session->turn_count;
    view->message_history_generation = session->message_history_generation;
    view->transcript_generation = session->transcript_generation;
    view->policy = session->policy;
    view->reasoning_policy = session->reasoning_policy;
    view->policy_set = session->policy_set;
    yvex_runtime_identity_copy(view->last_turn_identity,
                               session->last_turn_identity);
    yvex_runtime_identity_copy(view->state_digest, session->state_digest);
    yvex_runtime_identity_copy(view->generated_token_identity,
                               session->generated_token_identity);
    yvex_runtime_identity_copy(view->generated_text_digest,
                               session->generated_text_digest);
}

static int session_store_checkpoint_capture(
    server_session *session, server_session_store_view *view, yvex_error *err)
{
    int rc = YVEX_OK;
    if (session->generation) {
        rc = yvex_runtime_generation_context_checkpoint(
            session->generation, &view->generation_checkpoint, err);
        if (rc == YVEX_OK) view->generation_checkpoint_present = 1;
    } else if (session->pending_generation_checkpoint_present) {
        view->generation_checkpoint = session->pending_generation_checkpoint;
        view->generation_checkpoint_present = 1;
    }
    if (rc == YVEX_OK && session->committed_count &&
        !view->generation_checkpoint_present)
        rc = session_store_refuse(
            err, YVEX_ERR_STATE,
            "committed session state has no reproducible generation checkpoint");
    return rc;
}

int yvex_server_session_state_save(
    server_session *session, const char *path,
    yvex_runtime_state_store_summary *summary, yvex_error *err)
{
    server_session_store_view view;
    unsigned char *payload = NULL;
    unsigned long long payload_bytes = 0ull;
    char payload_identity[YVEX_SHA256_HEX_CAP];
    int rc;
    if (!session || !session->execution || !path || !summary)
        return session_store_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "session checkpoint save is invalid");
    session_store_view_set(session, &view);
    rc = session_store_checkpoint_capture(session, &view, err);
    if (rc == YVEX_OK)
        rc = yvex_server_session_store_encode(
            &view, &payload, &payload_bytes, payload_identity, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_state_save(
            session->execution, path, payload, payload_bytes, summary, err);
    yvex_server_session_store_bytes_close(&payload);
    return rc;
}

static int session_store_state_fits(
    const server_session *session, const server_session_store_state *state)
{
    unsigned long long index, offset = 0ull;
    if (!session || !state || state->message_count > SESSION_MAX_MESSAGES ||
        state->transcript_count > session->transcript_capacity ||
        state->committed_count > session->token_capacity ||
        (state->message_count && (!state->messages || !state->transcript)) ||
        (state->committed_count && !state->committed_tokens))
        return 0;
    for (index = 0ull; index < state->message_count; ++index) {
        const unsigned char *content =
            (const unsigned char *)state->messages[index].content;
        const unsigned char *reasoning =
            (const unsigned char *)state->messages[index].reasoning_content;
        unsigned long long next;
        if (content != state->transcript + offset ||
            !yvex_core_u64_add(offset, state->messages[index].content_len + 1ull,
                               &next) ||
            next > state->transcript_count)
            return 0;
        offset = next;
        if (reasoning != state->transcript + offset ||
            !yvex_core_u64_add(
                offset, state->messages[index].reasoning_content_len + 1ull,
                &next) ||
            next > state->transcript_count)
            return 0;
        offset = next;
    }
    return offset == state->transcript_count;
}

static void session_store_state_apply(
    server_session *session, const server_session_store_state *state)
{
    unsigned long long index, offset = 0ull;
    memset(session->messages, 0, sizeof(session->messages));
    memset(session->transcript, 0, (size_t)session->transcript_capacity);
    memset(session->committed_tokens, 0,
           (size_t)session->token_capacity * sizeof(*session->committed_tokens));
    memset(session->prompt_tokens, 0,
           (size_t)session->token_capacity * sizeof(*session->prompt_tokens));
    memset(session->turn_text, 0, (size_t)session->text_capacity + 1u);
    if (state->transcript_count)
        memcpy(session->transcript, state->transcript,
               (size_t)state->transcript_count);
    for (index = 0ull; index < state->message_count; ++index) {
        session->messages[index].schema_version =
            YVEX_PROMPT_MESSAGE_SCHEMA_V1;
        session->messages[index].role = state->messages[index].role;
        session->messages[index].content =
            (const char *)session->transcript + offset;
        session->messages[index].content_len =
            state->messages[index].content_len;
        offset += state->messages[index].content_len + 1ull;
        session->messages[index].reasoning_content =
            (const char *)session->transcript + offset;
        session->messages[index].reasoning_content_len =
            state->messages[index].reasoning_content_len;
        offset += state->messages[index].reasoning_content_len + 1ull;
    }
    if (state->committed_count)
        memcpy(session->committed_tokens, state->committed_tokens,
               (size_t)state->committed_count *
                   sizeof(*session->committed_tokens));
    session->message_count = state->message_count;
    session->transcript_count = state->transcript_count;
    session->committed_count = state->committed_count;
    session->turn_count = state->turn_count;
    session->message_history_generation = state->message_history_generation;
    session->transcript_generation = state->transcript_generation;
    session->policy = state->policy;
    session->reasoning_policy = state->reasoning_policy;
    session->policy_set = state->policy_set;
    session->pending_generation_checkpoint = state->generation_checkpoint;
    session->pending_generation_checkpoint_present =
        state->generation_checkpoint_present;
    yvex_runtime_identity_copy(session->last_turn_identity,
                               state->last_turn_identity);
    yvex_runtime_identity_copy(session->state_digest, state->state_digest);
    yvex_runtime_identity_copy(session->generated_token_identity,
                               state->generated_token_identity);
    yvex_runtime_identity_copy(session->generated_text_digest,
                               state->generated_text_digest);
    memset(&session->partial_turn, 0, sizeof(session->partial_turn));
    atomic_store_explicit(&session->cancel_requested, 0, memory_order_release);
    session->state = session->attached_clients ? YVEX_SERVER_SESSION_READY
                                               : YVEX_SERVER_SESSION_DETACHED;
}

int yvex_server_session_state_clone(
    server_session *source, server_session *destination,
    unsigned long long vocabulary_size, yvex_error *err)
{
    server_session_store_view view;
    server_session_store_state state = {0};
    unsigned char *payload = NULL;
    unsigned long long payload_bytes = 0ull;
    char payload_identity[YVEX_SHA256_HEX_CAP];
    int rc;
    if (!source || !destination || source == destination || !vocabulary_size ||
        destination->generation || destination->message_count ||
        destination->committed_count || destination->transcript_count)
        return session_store_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "session clone endpoints are invalid");
    session_store_view_set(source, &view);
    rc = session_store_checkpoint_capture(source, &view, err);
    if (rc == YVEX_OK)
        rc = yvex_server_session_store_encode(
            &view, &payload, &payload_bytes, payload_identity, err);
    if (rc == YVEX_OK)
        rc = yvex_server_session_store_decode(
            payload, payload_bytes, SESSION_MAX_MESSAGES,
            destination->transcript_capacity, destination->token_capacity,
            vocabulary_size, &state, err);
    if (rc == YVEX_OK &&
        (state.committed_count != source->committed_count ||
         !session_store_state_fits(destination, &state)))
        rc = session_store_refuse(
            err, YVEX_ERR_FORMAT,
            "cloned semantic session extent diverged from its source");
    if (rc == YVEX_OK) session_store_state_apply(destination, &state);
    yvex_server_session_store_close(&state);
    yvex_server_session_store_bytes_close(&payload);
    return rc;
}

int yvex_server_session_state_restore(
    server_session *session, const char *path,
    unsigned long long maximum_file_bytes, unsigned long long vocabulary_size,
    yvex_runtime_state_store_summary *summary, yvex_error *err)
{
    yvex_runtime_state_store_payload payload = {0};
    yvex_runtime_state_store_summary inspected;
    server_session_store_state state = {0};
    int rc;
    if (!session || !session->execution || !path || !maximum_file_bytes ||
        !vocabulary_size || !summary)
        return session_store_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "session checkpoint restore is invalid");
    rc = yvex_runtime_session_state_inspect(
        session->execution, path, maximum_file_bytes, &payload, &inspected, err);
    if (rc == YVEX_OK &&
        (inspected.schema_version != YVEX_RUNTIME_STATE_STORE_SCHEMA_V2 ||
         !payload.byte_count))
        rc = session_store_refuse(
            err, YVEX_ERR_FORMAT,
            "state checkpoint lacks semantic session state");
    if (rc == YVEX_OK)
        rc = yvex_server_session_store_decode(
            payload.bytes, payload.byte_count, SESSION_MAX_MESSAGES,
            session->transcript_capacity, session->token_capacity,
            vocabulary_size, &state, err);
    if (rc == YVEX_OK &&
        (state.committed_count != inspected.committed_sequence_length ||
         !session_store_state_fits(session, &state)))
        rc = session_store_refuse(
            err, YVEX_ERR_FORMAT,
            "semantic and physical session checkpoint extents diverged");
    if (rc == YVEX_OK)
        rc = yvex_runtime_generation_context_close(&session->generation, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_state_restore(
            session->execution, path, maximum_file_bytes,
            state.committed_count, inspected.file_digest,
            inspected.payload_identity, summary, err);
    if (rc == YVEX_OK) session_store_state_apply(session, &state);
    yvex_server_session_store_close(&state);
    yvex_runtime_state_store_payload_close(&payload);
    return rc;
}

int yvex_server_session_generation_state_restore(
    server_session *session, yvex_error *err)
{
    int rc;
    if (!session)
        return session_store_refuse(err, YVEX_ERR_INVALID_ARG,
                                    "session generation restore is invalid");
    if (!session->pending_generation_checkpoint_present) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (!session->generation)
        return session_store_refuse(
            err, YVEX_ERR_STATE,
            "generation checkpoint has no open generation context");
    rc = yvex_runtime_generation_context_restore(
        session->generation, &session->pending_generation_checkpoint, err);
    if (rc == YVEX_OK) {
        memset(&session->pending_generation_checkpoint, 0,
               sizeof(session->pending_generation_checkpoint));
        session->pending_generation_checkpoint_present = 0;
    }
    return rc;
}
