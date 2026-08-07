/*
 * ggml_gguf_check.cpp - pinned official ggml GGUF structural checker.
 *
 * Owner: tests/external.
 * Owns: independent pinned-reader invocation and target-scale structural facts.
 * Does not own: YVEX parsing, payload validation, artifact admission, runtime,
 *   generation, upstream source code, or repository claims.
 * Invariants: this source is compiled only against exact ggml af97976; it reads
 *   metadata/tensor structure with no tensor allocation and never writes input.
 * Boundary: official structural parsing is not external runtime support.
 */
#include <ggml.h>
#include <gguf.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

static constexpr const char * PINNED_REVISION =
    "af97976c7810cdabb1863172f31c432dab767de7";

static bool parse_u64(const char * text, uint64_t * out) {
    char * end = nullptr;
    unsigned long long value;
    if (!text || !text[0] || text[0] == '-') return false;
    errno = 0;
    value = std::strtoull(text, &end, 10);
    if (errno || !end || *end) return false;
    *out = value;
    return true;
}

static bool required_tokenizer(const gguf_context * ctx,
                               uint64_t tokens,
                               uint64_t merges) {
    int64_t token_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    int64_t type_key = gguf_find_key(ctx, "tokenizer.ggml.token_type");
    int64_t merge_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    return token_key >= 0 && type_key >= 0 && merge_key >= 0 &&
           gguf_get_kv_type(ctx, token_key) == GGUF_TYPE_ARRAY &&
           gguf_get_arr_type(ctx, token_key) == GGUF_TYPE_STRING &&
           gguf_get_arr_n(ctx, token_key) == tokens &&
           gguf_get_kv_type(ctx, type_key) == GGUF_TYPE_ARRAY &&
           gguf_get_arr_type(ctx, type_key) == GGUF_TYPE_INT32 &&
           gguf_get_arr_n(ctx, type_key) == tokens &&
           gguf_get_kv_type(ctx, merge_key) == GGUF_TYPE_ARRAY &&
           gguf_get_arr_type(ctx, merge_key) == GGUF_TYPE_STRING &&
           gguf_get_arr_n(ctx, merge_key) == merges &&
           gguf_find_key(ctx, "tokenizer.huggingface.json") >= 0 &&
           gguf_find_key(ctx, "yvex.tokenizer.config.json") >= 0;
}

static bool admitted_payload_class(const gguf_context * ctx,
                                   uint64_t tokens,
                                   uint64_t merges) {
    if (tokens || merges) return required_tokenizer(ctx, tokens, merges);
    int64_t target_key = gguf_find_key(ctx, "yvex.logical.target");
    int64_t component_key = gguf_find_key(ctx, "yvex.logical.component");
    int64_t identity_key =
        gguf_find_key(ctx, "yvex.logical.component.identity");
    return target_key >= 0 && component_key >= 0 && identity_key >= 0 &&
           gguf_get_kv_type(ctx, target_key) == GGUF_TYPE_STRING &&
           gguf_get_kv_type(ctx, component_key) == GGUF_TYPE_STRING &&
           gguf_get_kv_type(ctx, identity_key) == GGUF_TYPE_STRING &&
           std::strlen(gguf_get_val_str(ctx, identity_key)) == 64u;
}

int main(int argc, char ** argv) {
    uint64_t expected_size;
    uint64_t expected_metadata;
    uint64_t expected_tensors;
    uint64_t expected_tokens;
    uint64_t expected_merges;
    struct stat st {};
    ggml_context * tensor_ctx = nullptr;
    gguf_init_params params {};
    gguf_context * gguf;
    uint64_t raw_bytes = 0;
    uint64_t padded_end = 0;

    if (argc != 7 || !parse_u64(argv[2], &expected_size) ||
        !parse_u64(argv[3], &expected_metadata) ||
        !parse_u64(argv[4], &expected_tensors) ||
        !parse_u64(argv[5], &expected_tokens) ||
        !parse_u64(argv[6], &expected_merges)) {
        std::fprintf(stderr,
                     "usage: %s ARTIFACT SIZE METADATA TENSORS TOKENS MERGES\n",
                     argv[0]);
        return 2;
    }
    if (stat(argv[1], &st) != 0 || !S_ISREG(st.st_mode) ||
        static_cast<uint64_t>(st.st_size) != expected_size) {
        std::fprintf(stderr, "official_file_size=refused\n");
        return 1;
    }
    params.no_alloc = true;
    params.ctx = &tensor_ctx;
    gguf = gguf_init_from_file(argv[1], params);
    if (!gguf || !tensor_ctx) {
        std::fprintf(stderr, "official_reader=refused\n");
        if (gguf) gguf_free(gguf);
        if (tensor_ctx) ggml_free(tensor_ctx);
        return 1;
    }
    if (gguf_get_version(gguf) != 3u ||
        gguf_get_n_kv(gguf) != static_cast<int64_t>(expected_metadata) ||
        gguf_get_n_tensors(gguf) != static_cast<int64_t>(expected_tensors) ||
        gguf_get_alignment(gguf) != 32u ||
        !admitted_payload_class(gguf, expected_tokens, expected_merges)) {
        std::fprintf(stderr, "official_structure=refused\n");
        gguf_free(gguf);
        ggml_free(tensor_ctx);
        return 1;
    }
    for (uint64_t i = 0; i < expected_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf, static_cast<int64_t>(i));
        ggml_tensor * tensor = name ? ggml_get_tensor(tensor_ctx, name) : nullptr;
        size_t offset = gguf_get_tensor_offset(gguf, static_cast<int64_t>(i));
        size_t size = gguf_get_tensor_size(gguf, static_cast<int64_t>(i));
        if (!name || !name[0] || !tensor || tensor->type !=
                gguf_get_tensor_type(gguf, static_cast<int64_t>(i)) ||
            ggml_n_dims(tensor) < 1 || ggml_n_dims(tensor) > 4 ||
            offset != padded_end || size != ggml_nbytes(tensor) ||
            raw_bytes > UINT64_MAX - size || offset > SIZE_MAX - size) {
            std::fprintf(stderr, "official_tensor=%" PRIu64 " refused\n", i);
            gguf_free(gguf);
            ggml_free(tensor_ctx);
            return 1;
        }
        raw_bytes += size;
        padded_end = (offset + size + 31u) & ~static_cast<uint64_t>(31u);
    }
    if (gguf_get_data_offset(gguf) > expected_size ||
        padded_end > expected_size - gguf_get_data_offset(gguf) ||
        gguf_get_data_offset(gguf) + padded_end != expected_size) {
        std::fprintf(stderr, "official_data_span=refused\n");
        gguf_free(gguf);
        ggml_free(tensor_ctx);
        return 1;
    }
    std::printf("official_revision=%s\n", PINNED_REVISION);
    std::printf("official_version=%u\n", gguf_get_version(gguf));
    std::printf("official_metadata=%" PRId64 "\n", gguf_get_n_kv(gguf));
    std::printf("official_tensors=%" PRId64 "\n", gguf_get_n_tensors(gguf));
    std::printf("official_data_offset=%zu\n", gguf_get_data_offset(gguf));
    std::printf("official_raw_tensor_bytes=%" PRIu64 "\n", raw_bytes);
    std::printf("official_file_bytes=%" PRIu64 "\n", expected_size);
    std::printf("official_result=accepted\n");
    gguf_free(gguf);
    ggml_free(tensor_ctx);
    return 0;
}
