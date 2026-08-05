/*
 * Keep one tokenizer object while separating its admitted algorithms and mutable decoders.
 *
 * Only tokenizer translation units include this header; borrowed token text follows tokenizer
 * lifetime. Source-local ABI shared by metadata, execution, and decoder owners.
 */
#ifndef SRC_TOKENIZER_PRIVATE_H_INCLUDED
#define SRC_TOKENIZER_PRIVATE_H_INCLUDED

#include <stdint.h>
#include <yvex/gguf.h>
#include <yvex/internal/conversation.h>
#include <yvex/tokenizer.h>

typedef struct {
    int present;
    unsigned int id;
} tokenizer_special_id;

typedef struct {
    uint64_t hash;
    unsigned int id;
    int occupied;
} tokenizer_vocab_slot;

typedef struct {
    uint64_t pair;
    unsigned int rank, merged_id;
    int occupied;
} tokenizer_merge_slot;

struct yvex_tokenizer {
    yvex_tokenizer_kind kind;
    yvex_tokenizer_support support;
    char *model_name;
    yvex_token_info *tokens;
    unsigned long long vocab_size;
    tokenizer_special_id bos, eos, unk, pad, sep;
    char *chat_template;
    unsigned long long chat_template_len;
    int has_huggingface_json;

    const yvex_conversation_protocol *conversation;
    yvex_tokenizer_plan_summary plan;
    tokenizer_vocab_slot *vocab_index;
    unsigned long long vocab_index_capacity;
    tokenizer_merge_slot *merge_index;
    unsigned long long merge_index_capacity;
    unsigned int *added_token_ids;
    unsigned long long added_token_count;
    unsigned long long added_bucket_start[257];
    unsigned int byte_token_ids[256];
};

enum {
    TOKENIZER_UNICODE_NUMBER = 1u << 0,
    TOKENIZER_UNICODE_LETTER_MARK = 1u << 1,
    TOKENIZER_UNICODE_PUNCT_SYMBOL = 1u << 2,
    TOKENIZER_UNICODE_SPACE = 1u << 3
};

int yvex_tokenizer_utf8_next(const unsigned char *bytes, unsigned long long count,
                             unsigned long long *offset, uint32_t *point);
unsigned int yvex_tokenizer_unicode_class(uint32_t point);

int yvex_tokenizer_execution_seal(yvex_tokenizer *tokenizer, const yvex_gguf *gguf,
                                  const yvex_model_descriptor *model, yvex_error *err);
void yvex_tokenizer_execution_release(yvex_tokenizer *tokenizer);
#endif
