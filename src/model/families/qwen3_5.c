/* Parse and seal Qwen3.5-family semantics from one authenticated source revision. */
#include <yvex/internal/families/qwen3_5.h>

#include <yvex/internal/core.h>
#include <yvex/internal/source_catalog.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_CONFIG_CAP (1024u * 1024u)

typedef enum {
    QWEN_FIELD_U64 = 1,
    QWEN_FIELD_BOOL,
    QWEN_FIELD_TEXT,
    QWEN_FIELD_DOUBLE,
    QWEN_FIELD_NULL,
    QWEN_FIELD_LAYERS,
    QWEN_FIELD_ROPE,
    QWEN_FIELD_EMPTY_U64_ARRAY
} qwen_field_kind;

typedef struct {
    const char *key;
    unsigned long long bit;
    qwen_field_kind kind;
    size_t offset, capacity;
} qwen_field;

typedef struct {
    unsigned long long seen;
    yvex_qwen3_5_text_architecture *text;
} qwen_parse_state;

struct yvex_qwen3_5_model {
    yvex_qwen3_5_architecture architecture;
};

static int qwen_refuse(yvex_qwen3_5_failure *failure,
                       yvex_qwen3_5_failure_code code,
                       const char *field, const char *reason,
                       yvex_status status, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->reason = reason;
        yvex_core_text_copy(failure->field, sizeof(failure->field), field);
    }
    yvex_error_set(err, status, "qwen3_5.model", reason);
    return status;
}

static int qwen_json_double(yvex_json *json, double *out)
{
    char value[48], *end = NULL;
    double parsed;

    if (!out || !yvex_json_number_text(json, value, sizeof(value))) return 0;
    errno = 0;
    parsed = strtod(value, &end);
    if (errno || !end || *end || !isfinite(parsed)) return 0;
    *out = parsed;
    return 1;
}

static int qwen_json_null(yvex_json *json)
{
    if (!json) return 0;
    yvex_json_space(json);
    if ((size_t)(json->end - json->cursor) < 4u ||
        memcmp(json->cursor, "null", 4u) != 0)
        return 0;
    json->cursor += 4u;
    return 1;
}

static int qwen_json_empty_array(yvex_json *json)
{
    yvex_json_iter iter;

    return yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY) &&
           yvex_json_array_value(&iter) == YVEX_JSON_ITEM_END &&
           !iter.trailing_separator;
}

static int qwen_parse_layer_types(yvex_json *json,
                                  yvex_qwen3_5_text_architecture *text)
{
    char name[32];
    yvex_json_iter iter;
    yvex_json_item item;
    unsigned long long count = 0ull;

    if (!text || !yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY))
        return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (count >= YVEX_QWEN3_5_LAYER_CAP ||
            !yvex_json_string(json, name, sizeof(name)))
            return 0;
        if (strcmp(name, "linear_attention") == 0) {
            text->layers[count] = YVEX_QWEN3_5_LAYER_LINEAR_ATTENTION;
            text->linear_attention_layers++;
        } else if (strcmp(name, "full_attention") == 0) {
            text->layers[count] = YVEX_QWEN3_5_LAYER_FULL_ATTENTION;
            text->full_attention_layers++;
        } else {
            return 0;
        }
        count++;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           count != 0ull;
}

static int qwen_parse_rope(yvex_json *json,
                           yvex_qwen3_5_text_architecture *text)
{
    enum {
        ROPE_INTERLEAVED = 1u << 0,
        ROPE_SECTION = 1u << 1,
        ROPE_PARTIAL = 1u << 2,
        ROPE_THETA = 1u << 3,
        ROPE_TYPE = 1u << 4,
        ROPE_REQUIRED = (1u << 5) - 1u
    };
    char key[YVEX_JSON_KEY_CAP], type[32];
    yvex_json_iter iter;
    yvex_json_item item;
    unsigned long long seen = 0ull, count = 0ull, bit;

    if (!text || !yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "mrope_interleaved") == 0) {
            bit = ROPE_INTERLEAVED;
            if (!yvex_json_bool(json, &text->mrope_interleaved)) return 0;
        } else if (strcmp(key, "mrope_section") == 0) {
            bit = ROPE_SECTION;
            if (!yvex_json_u64_array(json, text->mrope_sections, 3u, &count) ||
                count != 3ull)
                return 0;
        } else if (strcmp(key, "partial_rotary_factor") == 0) {
            bit = ROPE_PARTIAL;
            if (!qwen_json_double(json, &text->rope_partial_rotary_factor)) return 0;
        } else if (strcmp(key, "rope_theta") == 0) {
            bit = ROPE_THETA;
            if (!yvex_json_u64(json, &text->rope_theta)) return 0;
        } else if (strcmp(key, "rope_type") == 0) {
            bit = ROPE_TYPE;
            if (!yvex_json_string(json, type, sizeof(type)) ||
                strcmp(type, "default") != 0)
                return 0;
        } else {
            if (!yvex_json_skip_value(json)) return 0;
            continue;
        }
        if (seen & bit) return 0;
        seen |= bit;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           seen == ROPE_REQUIRED;
}

static int qwen_field_apply(yvex_json *json, const qwen_field *field,
                            void *base, qwen_parse_state *state)
{
    unsigned char *bytes = base;

    switch (field->kind) {
    case QWEN_FIELD_U64:
        return yvex_json_u64(
            json, (unsigned long long *)(void *)(bytes + field->offset));
    case QWEN_FIELD_BOOL:
        return yvex_json_bool(json, (int *)(void *)(bytes + field->offset));
    case QWEN_FIELD_TEXT:
        return yvex_json_string(json, (char *)(bytes + field->offset),
                                field->capacity);
    case QWEN_FIELD_DOUBLE:
        return qwen_json_double(json, (double *)(void *)(bytes + field->offset));
    case QWEN_FIELD_NULL:
        return qwen_json_null(json);
    case QWEN_FIELD_LAYERS:
        return qwen_parse_layer_types(json, state->text);
    case QWEN_FIELD_ROPE:
        return qwen_parse_rope(json, state->text);
    case QWEN_FIELD_EMPTY_U64_ARRAY:
        return qwen_json_empty_array(json);
    }
    return 0;
}

static int qwen_object_parse(yvex_json *json, const qwen_field *fields,
                             size_t field_count, unsigned long long required,
                             void *base, qwen_parse_state *state)
{
    char key[YVEX_JSON_KEY_CAP];
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        const qwen_field *field = yvex_core_keyed_row_find(
            fields, field_count, sizeof(fields[0]), offsetof(qwen_field, key), key);

        if (!field) {
            if (!yvex_json_skip_value(json)) return 0;
        } else if ((state->seen & field->bit) ||
                   !qwen_field_apply(json, field, base, state)) {
            return 0;
        } else {
            state->seen |= field->bit;
        }
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           (state->seen & required) == required;
}

#define QWEN_TEXT_FIELD(bit, name, kind, member) \
    {name, 1ull << bit, kind, offsetof(yvex_qwen3_5_text_architecture, member), \
     sizeof(((yvex_qwen3_5_text_architecture *)0)->member)}

static const qwen_field qwen_text_fields[] = {
    QWEN_TEXT_FIELD(0, "attention_bias", QWEN_FIELD_BOOL, attention_bias),
    QWEN_TEXT_FIELD(1, "attention_dropout", QWEN_FIELD_DOUBLE, attention_dropout),
    QWEN_TEXT_FIELD(2, "attn_output_gate", QWEN_FIELD_BOOL, attention_output_gate),
    QWEN_TEXT_FIELD(3, "bos_token_id", QWEN_FIELD_U64, bos_token_id),
    QWEN_TEXT_FIELD(4, "dtype", QWEN_FIELD_TEXT, source_dtype),
    QWEN_TEXT_FIELD(5, "eos_token_id", QWEN_FIELD_U64, eos_token_id),
    QWEN_TEXT_FIELD(6, "full_attention_interval", QWEN_FIELD_U64,
                    full_attention_interval),
    QWEN_TEXT_FIELD(7, "head_dim", QWEN_FIELD_U64, attention_head_dimension),
    QWEN_TEXT_FIELD(8, "hidden_act", QWEN_FIELD_TEXT, hidden_activation),
    QWEN_TEXT_FIELD(9, "hidden_size", QWEN_FIELD_U64, hidden_size),
    QWEN_TEXT_FIELD(10, "intermediate_size", QWEN_FIELD_U64, intermediate_size),
    {"layer_types", 1ull << 11, QWEN_FIELD_LAYERS, 0u, 0u},
    QWEN_TEXT_FIELD(12, "linear_conv_kernel_dim", QWEN_FIELD_U64,
                    linear_convolution_kernel),
    QWEN_TEXT_FIELD(13, "linear_key_head_dim", QWEN_FIELD_U64,
                    linear_key_head_dimension),
    QWEN_TEXT_FIELD(14, "linear_num_key_heads", QWEN_FIELD_U64,
                    linear_key_heads),
    QWEN_TEXT_FIELD(15, "linear_num_value_heads", QWEN_FIELD_U64,
                    linear_value_heads),
    QWEN_TEXT_FIELD(16, "linear_value_head_dim", QWEN_FIELD_U64,
                    linear_value_head_dimension),
    QWEN_TEXT_FIELD(17, "mamba_ssm_dtype", QWEN_FIELD_TEXT,
                    recurrent_state_dtype),
    QWEN_TEXT_FIELD(18, "max_position_embeddings", QWEN_FIELD_U64,
                    maximum_positions),
    QWEN_TEXT_FIELD(19, "model_type", QWEN_FIELD_TEXT, model_type),
    QWEN_TEXT_FIELD(20, "mtp_num_hidden_layers", QWEN_FIELD_U64,
                    mtp_hidden_layers),
    QWEN_TEXT_FIELD(21, "mtp_use_dedicated_embeddings", QWEN_FIELD_BOOL,
                    mtp_dedicated_embeddings),
    QWEN_TEXT_FIELD(22, "num_attention_heads", QWEN_FIELD_U64, attention_heads),
    QWEN_TEXT_FIELD(23, "num_hidden_layers", QWEN_FIELD_U64, layer_count),
    QWEN_TEXT_FIELD(24, "num_key_value_heads", QWEN_FIELD_U64, kv_heads),
    QWEN_TEXT_FIELD(25, "output_gate_type", QWEN_FIELD_TEXT, output_gate_type),
    {"pad_token_id", 1ull << 26, QWEN_FIELD_NULL, 0u, 0u},
    QWEN_TEXT_FIELD(27, "partial_rotary_factor", QWEN_FIELD_DOUBLE,
                    partial_rotary_factor),
    QWEN_TEXT_FIELD(28, "rms_norm_eps", QWEN_FIELD_DOUBLE, rms_norm_epsilon),
    {"rope_parameters", 1ull << 29, QWEN_FIELD_ROPE, 0u, 0u},
    QWEN_TEXT_FIELD(30, "tie_word_embeddings", QWEN_FIELD_BOOL, tied_embeddings),
    QWEN_TEXT_FIELD(31, "use_cache", QWEN_FIELD_BOOL, use_cache),
    QWEN_TEXT_FIELD(32, "vocab_size", QWEN_FIELD_U64, vocabulary_size),
};

#define QWEN_TEXT_REQUIRED ((1ull << 33) - 1ull)

static int qwen_parse_text_config(yvex_json *json,
                                  yvex_qwen3_5_text_architecture *text)
{
    qwen_parse_state state = {0};

    state.text = text;
    return qwen_object_parse(
        json, qwen_text_fields,
        sizeof(qwen_text_fields) / sizeof(qwen_text_fields[0]),
        QWEN_TEXT_REQUIRED, text, &state);
}

#define QWEN_VISION_FIELD(bit, name, kind, member) \
    {name, 1ull << bit, kind, offsetof(yvex_qwen3_5_vision_architecture, member), \
     sizeof(((yvex_qwen3_5_vision_architecture *)0)->member)}

static const qwen_field qwen_vision_fields[] = {
    {"deepstack_visual_indexes", 1ull << 0, QWEN_FIELD_EMPTY_U64_ARRAY,
     offsetof(yvex_qwen3_5_vision_architecture, deepstack_index_count),
     sizeof(unsigned long long)},
    QWEN_VISION_FIELD(1, "depth", QWEN_FIELD_U64, depth),
    QWEN_VISION_FIELD(2, "hidden_act", QWEN_FIELD_TEXT, hidden_activation),
    QWEN_VISION_FIELD(3, "hidden_size", QWEN_FIELD_U64, hidden_size),
    QWEN_VISION_FIELD(4, "intermediate_size", QWEN_FIELD_U64, intermediate_size),
    QWEN_VISION_FIELD(5, "model_type", QWEN_FIELD_TEXT, model_type),
    QWEN_VISION_FIELD(6, "num_heads", QWEN_FIELD_U64, heads),
    QWEN_VISION_FIELD(7, "num_position_embeddings", QWEN_FIELD_U64,
                      position_count),
    QWEN_VISION_FIELD(8, "out_hidden_size", QWEN_FIELD_U64,
                      output_hidden_size),
    QWEN_VISION_FIELD(9, "patch_size", QWEN_FIELD_U64, patch_size),
    QWEN_VISION_FIELD(10, "spatial_merge_size", QWEN_FIELD_U64,
                      spatial_merge_size),
    QWEN_VISION_FIELD(11, "temporal_patch_size", QWEN_FIELD_U64,
                      temporal_patch_size),
};

static int qwen_parse_vision_config(yvex_json *json,
                                    yvex_qwen3_5_vision_architecture *vision)
{
    qwen_parse_state state = {0};

    return qwen_object_parse(
        json, qwen_vision_fields,
        sizeof(qwen_vision_fields) / sizeof(qwen_vision_fields[0]),
        (1ull << 12) - 1ull, vision, &state);
}

static int qwen_parse_architectures(yvex_json *json)
{
    char name[64];
    yvex_json_iter iter;
    yvex_json_item item;
    unsigned long long count = 0ull;
    int matched = 0;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY)) return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (!yvex_json_string(json, name, sizeof(name))) return 0;
        if (strcmp(name, "Qwen3_5ForConditionalGeneration") == 0) matched = 1;
        count++;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           count != 0ull && matched;
}

static int qwen_parse_config(const char *data, size_t length,
                             yvex_qwen3_5_architecture *architecture)
{
    enum {
        OUTER_ARCHITECTURES = 1u << 0,
        OUTER_IMAGE_TOKEN = 1u << 1,
        OUTER_LANGUAGE_ONLY = 1u << 2,
        OUTER_MODEL_TYPE = 1u << 3,
        OUTER_TEXT_CONFIG = 1u << 4,
        OUTER_TIED = 1u << 5,
        OUTER_TRANSFORMERS = 1u << 6,
        OUTER_VIDEO_TOKEN = 1u << 7,
        OUTER_VISION_CONFIG = 1u << 8,
        OUTER_VISION_END = 1u << 9,
        OUTER_VISION_START = 1u << 10,
        OUTER_REQUIRED = (1u << 11) - 1u
    };
    char key[YVEX_JSON_KEY_CAP], model_type[32];
    yvex_json json;
    yvex_json_iter iter;
    yvex_json_item item;
    unsigned long long seen = 0ull, bit;
    int language_only = 0, tied = 0;

    yvex_json_init(&json, data, length);
    if (!yvex_json_iter_begin(&json, &iter, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "architectures") == 0) {
            bit = OUTER_ARCHITECTURES;
            if (!qwen_parse_architectures(&json)) return 0;
        } else if (strcmp(key, "image_token_id") == 0) {
            bit = OUTER_IMAGE_TOKEN;
            if (!yvex_json_u64(&json, &architecture->vision.image_token_id)) return 0;
        } else if (strcmp(key, "language_model_only") == 0) {
            bit = OUTER_LANGUAGE_ONLY;
            if (!yvex_json_bool(&json, &language_only)) return 0;
        } else if (strcmp(key, "model_type") == 0) {
            bit = OUTER_MODEL_TYPE;
            if (!yvex_json_string(&json, model_type, sizeof(model_type)) ||
                strcmp(model_type, "qwen3_5") != 0)
                return 0;
        } else if (strcmp(key, "text_config") == 0) {
            bit = OUTER_TEXT_CONFIG;
            if (!qwen_parse_text_config(&json, &architecture->text)) return 0;
        } else if (strcmp(key, "tie_word_embeddings") == 0) {
            bit = OUTER_TIED;
            if (!yvex_json_bool(&json, &tied)) return 0;
        } else if (strcmp(key, "transformers_version") == 0) {
            bit = OUTER_TRANSFORMERS;
            if (!yvex_json_string(&json, architecture->transformers_version,
                                  sizeof(architecture->transformers_version)))
                return 0;
        } else if (strcmp(key, "video_token_id") == 0) {
            bit = OUTER_VIDEO_TOKEN;
            if (!yvex_json_u64(&json, &architecture->vision.video_token_id)) return 0;
        } else if (strcmp(key, "vision_config") == 0) {
            bit = OUTER_VISION_CONFIG;
            if (!qwen_parse_vision_config(&json, &architecture->vision)) return 0;
        } else if (strcmp(key, "vision_end_token_id") == 0) {
            bit = OUTER_VISION_END;
            if (!yvex_json_u64(&json, &architecture->vision.vision_end_token_id)) return 0;
        } else if (strcmp(key, "vision_start_token_id") == 0) {
            bit = OUTER_VISION_START;
            if (!yvex_json_u64(&json, &architecture->vision.vision_start_token_id)) return 0;
        } else {
            if (!yvex_json_skip_value(&json)) return 0;
            continue;
        }
        if (seen & bit) return 0;
        seen |= bit;
    }
    architecture->source_multimodal = !language_only;
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           yvex_json_complete(&json) && seen == OUTER_REQUIRED && !tied;
}

static int qwen_parse_stop_tokens(yvex_json *json,
                                  yvex_qwen3_5_generation_policy *policy)
{
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY)) return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (policy->stop_token_count >= YVEX_QWEN3_5_GENERATION_STOP_CAP ||
            !yvex_json_u64(json, &policy->stop_token_ids[policy->stop_token_count]))
            return 0;
        policy->stop_token_count++;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           policy->stop_token_count != 0ull;
}

static int qwen_parse_generation(const char *data, size_t length,
                                 yvex_qwen3_5_generation_policy *policy)
{
    enum {
        GEN_BOS = 1u << 0, GEN_SAMPLE = 1u << 1, GEN_EOS = 1u << 2,
        GEN_PAD = 1u << 3, GEN_TEMPERATURE = 1u << 4, GEN_TOP_K = 1u << 5,
        GEN_TOP_P = 1u << 6, GEN_REQUIRED = (1u << 7) - 1u
    };
    char key[YVEX_JSON_KEY_CAP];
    yvex_json json;
    yvex_json_iter iter;
    yvex_json_item item;
    unsigned long long seen = 0ull, bit;

    yvex_json_init(&json, data, length);
    if (!yvex_json_iter_begin(&json, &iter, YVEX_JSON_COLLECTION_OBJECT)) return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        if (strcmp(key, "bos_token_id") == 0) {
            bit = GEN_BOS;
            if (!yvex_json_u64(&json, &policy->bos_token_id)) return 0;
        } else if (strcmp(key, "do_sample") == 0) {
            bit = GEN_SAMPLE;
            if (!yvex_json_bool(&json, &policy->do_sample)) return 0;
        } else if (strcmp(key, "eos_token_id") == 0) {
            bit = GEN_EOS;
            if (!qwen_parse_stop_tokens(&json, policy)) return 0;
        } else if (strcmp(key, "pad_token_id") == 0) {
            bit = GEN_PAD;
            if (!yvex_json_u64(&json, &policy->pad_token_id)) return 0;
        } else if (strcmp(key, "temperature") == 0) {
            bit = GEN_TEMPERATURE;
            if (!qwen_json_double(&json, &policy->temperature)) return 0;
        } else if (strcmp(key, "top_k") == 0) {
            bit = GEN_TOP_K;
            if (!yvex_json_u64(&json, &policy->top_k)) return 0;
        } else if (strcmp(key, "top_p") == 0) {
            bit = GEN_TOP_P;
            if (!qwen_json_double(&json, &policy->top_p)) return 0;
        } else {
            if (!yvex_json_skip_value(&json)) return 0;
            continue;
        }
        if (seen & bit) return 0;
        seen |= bit;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           yvex_json_complete(&json) && seen == GEN_REQUIRED;
}

static int qwen_validate(yvex_qwen3_5_architecture *architecture,
                         yvex_qwen3_5_failure *failure, yvex_error *err)
{
    yvex_qwen3_5_text_architecture *text = &architecture->text;
    unsigned long long layer, section_sum, rotary;
    int eos_present = 0;

    if (!architecture->source_multimodal || !text->layer_count ||
        text->layer_count > YVEX_QWEN3_5_LAYER_CAP ||
        text->layer_count != text->linear_attention_layers +
                                 text->full_attention_layers ||
        !text->full_attention_interval)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_CONFIGURATION,
                           "layer_types", "hybrid layer topology is inconsistent",
                           YVEX_ERR_FORMAT, err);
    for (layer = 0ull; layer < text->layer_count; ++layer) {
        yvex_qwen3_5_layer_kind expected =
            (layer + 1ull) % text->full_attention_interval == 0ull
                ? YVEX_QWEN3_5_LAYER_FULL_ATTENTION
                : YVEX_QWEN3_5_LAYER_LINEAR_ATTENTION;

        if (text->layers[layer] != expected)
            return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_CONFIGURATION,
                               "layer_types", "hybrid attention interval disagrees with layer types",
                               YVEX_ERR_FORMAT, err);
    }
    rotary = (unsigned long long)llround(
        (double)text->attention_head_dimension * text->partial_rotary_factor);
    section_sum = text->mrope_sections[0] + text->mrope_sections[1] +
                  text->mrope_sections[2];
    if (strcmp(text->model_type, YVEX_QWEN3_5_TEXT_MODEL_TYPE) != 0 ||
        strcmp(text->source_dtype, "bfloat16") != 0 ||
        strcmp(text->recurrent_state_dtype, "float32") != 0 ||
        strcmp(text->hidden_activation, "silu") != 0 ||
        strcmp(text->output_gate_type, "swish") != 0 ||
        text->attention_bias || text->attention_dropout != 0.0 ||
        !text->attention_output_gate || !text->use_cache || text->tied_embeddings ||
        text->mtp_dedicated_embeddings || text->partial_rotary_factor <= 0.0 ||
        text->partial_rotary_factor != text->rope_partial_rotary_factor ||
        rotary == 0ull || section_sum * 2ull != rotary ||
        text->linear_key_heads == 0ull || text->linear_value_heads == 0ull ||
        text->linear_key_head_dimension == 0ull ||
        text->linear_value_head_dimension == 0ull ||
        text->linear_convolution_kernel == 0ull || text->attention_heads == 0ull ||
        text->kv_heads == 0ull || text->attention_heads % text->kv_heads != 0ull ||
        architecture->vision.output_hidden_size != text->hidden_size ||
        strcmp(architecture->vision.model_type, "qwen3_5") != 0)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_CONFIGURATION,
                           "text_config", "Qwen3.5 architecture invariants are inconsistent",
                           YVEX_ERR_FORMAT, err);
    text->rotary_dimension = rotary;
    text->recurrent_state_f32 = 1;
    for (layer = 0ull; layer < architecture->generation.stop_token_count; ++layer)
        if (architecture->generation.stop_token_ids[layer] == text->eos_token_id)
            eos_present = 1;
    if (architecture->generation.bos_token_id != text->bos_token_id ||
        architecture->generation.pad_token_id != text->eos_token_id || !eos_present ||
        architecture->generation.temperature <= 0.0 ||
        architecture->generation.top_p <= 0.0 ||
        architecture->generation.top_p > 1.0)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_GENERATION_POLICY,
                           "generation_config", "generation tokens or sampling policy disagree with the model",
                           YVEX_ERR_FORMAT, err);
    architecture->text_specialization = 1;
    architecture->vision_execution_deferred = 1;
    architecture->mtp_acceleration_deferred = text->mtp_hidden_layers != 0ull;
    return YVEX_OK;
}

typedef struct {
    const char *suffix;
    yvex_qwen3_5_tensor_role role;
} qwen_tensor_suffix;

static const qwen_tensor_suffix qwen_full_attention_tensors[] = {
    {"self_attn.q_proj.weight", YVEX_QWEN3_5_ROLE_ATTENTION_Q},
    {"self_attn.k_proj.weight", YVEX_QWEN3_5_ROLE_ATTENTION_K},
    {"self_attn.v_proj.weight", YVEX_QWEN3_5_ROLE_ATTENTION_V},
    {"self_attn.o_proj.weight", YVEX_QWEN3_5_ROLE_ATTENTION_OUT},
    {"self_attn.q_norm.weight", YVEX_QWEN3_5_ROLE_ATTENTION_Q_NORM},
    {"self_attn.k_norm.weight", YVEX_QWEN3_5_ROLE_ATTENTION_K_NORM}};

static const qwen_tensor_suffix qwen_delta_tensors[] = {
    {"linear_attn.A_log", YVEX_QWEN3_5_ROLE_DELTA_DECAY_LOG},
    {"linear_attn.conv1d.weight", YVEX_QWEN3_5_ROLE_DELTA_CONVOLUTION},
    {"linear_attn.dt_bias", YVEX_QWEN3_5_ROLE_DELTA_TIME_BIAS},
    {"linear_attn.in_proj_a.weight", YVEX_QWEN3_5_ROLE_DELTA_DECAY_PROJECTION},
    {"linear_attn.in_proj_b.weight", YVEX_QWEN3_5_ROLE_DELTA_BETA_PROJECTION},
    {"linear_attn.in_proj_qkv.weight", YVEX_QWEN3_5_ROLE_DELTA_QKV_PROJECTION},
    {"linear_attn.in_proj_z.weight", YVEX_QWEN3_5_ROLE_DELTA_OUTPUT_GATE},
    {"linear_attn.norm.weight", YVEX_QWEN3_5_ROLE_DELTA_OUTPUT_NORM},
    {"linear_attn.out_proj.weight", YVEX_QWEN3_5_ROLE_DELTA_OUTPUT}};

static const qwen_tensor_suffix qwen_common_layer_tensors[] = {
    {"input_layernorm.weight", YVEX_QWEN3_5_ROLE_INPUT_NORM},
    {"mlp.gate_proj.weight", YVEX_QWEN3_5_ROLE_FFN_GATE},
    {"mlp.up_proj.weight", YVEX_QWEN3_5_ROLE_FFN_UP},
    {"mlp.down_proj.weight", YVEX_QWEN3_5_ROLE_FFN_DOWN},
    {"post_attention_layernorm.weight", YVEX_QWEN3_5_ROLE_POST_ATTENTION_NORM}};

static int qwen_indexed_name(const char *name, const char *prefix,
                             unsigned long long *index, const char **suffix)
{
    const char *cursor;
    char *end = NULL;
    unsigned long long value;

    if (!name || !prefix || strncmp(name, prefix, strlen(prefix)) != 0) return 0;
    cursor = name + strlen(prefix);
    if (*cursor < '0' || *cursor > '9') return 0;
    errno = 0;
    value = strtoull(cursor, &end, 10);
    if (errno || !end || end == cursor || *end != '.' || !end[1]) return 0;
    *index = value;
    *suffix = end + 1;
    return 1;
}

static yvex_qwen3_5_tensor_role qwen_suffix_role(
    const char *suffix, const qwen_tensor_suffix *rows, size_t count)
{
    size_t index;

    for (index = 0u; index < count; ++index)
        if (strcmp(suffix, rows[index].suffix) == 0) return rows[index].role;
    return YVEX_QWEN3_5_ROLE_UNKNOWN;
}

static int qwen_bf16_shape(const yvex_native_weight_info *tensor,
                           unsigned int rank, const unsigned long long *dims)
{
    unsigned long long elements = 1ull;
    unsigned int index;

    if (!tensor || tensor->dtype != YVEX_NATIVE_DTYPE_BF16 ||
        tensor->rank != rank || !rank)
        return 0;
    for (index = 0u; index < rank; ++index) {
        if (!dims[index] || tensor->dims[index] != dims[index] ||
            elements > ULLONG_MAX / dims[index])
            return 0;
        elements *= dims[index];
    }
    return elements <= ULLONG_MAX / 2ull && tensor->data_bytes == elements * 2ull;
}

static int qwen_text_tensor_shape(
    const yvex_qwen3_5_text_architecture *text,
    const yvex_native_weight_info *tensor, yvex_qwen3_5_tensor_role role)
{
    unsigned long long dims[3] = {0ull, 0ull, 0ull};
    unsigned long long q_width = text->linear_key_heads * text->linear_key_head_dimension;
    unsigned long long v_width = text->linear_value_heads * text->linear_value_head_dimension;
    unsigned long long attention_width = text->attention_heads * text->attention_head_dimension;
    unsigned long long kv_width = text->kv_heads * text->attention_head_dimension;
    unsigned int rank = 1u;

    switch (role) {
    case YVEX_QWEN3_5_ROLE_TOKEN_EMBEDDING:
    case YVEX_QWEN3_5_ROLE_OUTPUT_HEAD:
        rank = 2u; dims[0] = text->vocabulary_size; dims[1] = text->hidden_size; break;
    case YVEX_QWEN3_5_ROLE_OUTPUT_NORM:
    case YVEX_QWEN3_5_ROLE_INPUT_NORM:
    case YVEX_QWEN3_5_ROLE_POST_ATTENTION_NORM:
        dims[0] = text->hidden_size; break;
    case YVEX_QWEN3_5_ROLE_FFN_GATE:
    case YVEX_QWEN3_5_ROLE_FFN_UP:
        rank = 2u; dims[0] = text->intermediate_size; dims[1] = text->hidden_size; break;
    case YVEX_QWEN3_5_ROLE_FFN_DOWN:
        rank = 2u; dims[0] = text->hidden_size; dims[1] = text->intermediate_size; break;
    case YVEX_QWEN3_5_ROLE_ATTENTION_Q:
        rank = 2u; dims[0] = attention_width * 2ull; dims[1] = text->hidden_size; break;
    case YVEX_QWEN3_5_ROLE_ATTENTION_K:
    case YVEX_QWEN3_5_ROLE_ATTENTION_V:
        rank = 2u; dims[0] = kv_width; dims[1] = text->hidden_size; break;
    case YVEX_QWEN3_5_ROLE_ATTENTION_OUT:
        rank = 2u; dims[0] = text->hidden_size; dims[1] = attention_width; break;
    case YVEX_QWEN3_5_ROLE_ATTENTION_Q_NORM:
    case YVEX_QWEN3_5_ROLE_ATTENTION_K_NORM:
        dims[0] = text->attention_head_dimension; break;
    case YVEX_QWEN3_5_ROLE_DELTA_DECAY_LOG:
    case YVEX_QWEN3_5_ROLE_DELTA_TIME_BIAS:
        dims[0] = text->linear_value_heads; break;
    case YVEX_QWEN3_5_ROLE_DELTA_CONVOLUTION:
        rank = 3u; dims[0] = q_width * 2ull + v_width; dims[1] = 1ull;
        dims[2] = text->linear_convolution_kernel; break;
    case YVEX_QWEN3_5_ROLE_DELTA_DECAY_PROJECTION:
    case YVEX_QWEN3_5_ROLE_DELTA_BETA_PROJECTION:
        rank = 2u; dims[0] = text->linear_value_heads; dims[1] = text->hidden_size; break;
    case YVEX_QWEN3_5_ROLE_DELTA_QKV_PROJECTION:
        rank = 2u; dims[0] = q_width * 2ull + v_width; dims[1] = text->hidden_size; break;
    case YVEX_QWEN3_5_ROLE_DELTA_OUTPUT_GATE:
        rank = 2u; dims[0] = v_width; dims[1] = text->hidden_size; break;
    case YVEX_QWEN3_5_ROLE_DELTA_OUTPUT_NORM:
        dims[0] = text->linear_value_head_dimension; break;
    case YVEX_QWEN3_5_ROLE_DELTA_OUTPUT:
        rank = 2u; dims[0] = text->hidden_size; dims[1] = v_width; break;
    default:
        return 0;
    }
    return qwen_bf16_shape(tensor, rank, dims);
}

static int qwen_deferred_tensor_valid(const yvex_native_weight_info *tensor)
{
    unsigned long long elements = 1ull;
    unsigned int index;

    if (!tensor || tensor->dtype != YVEX_NATIVE_DTYPE_BF16 || !tensor->rank) return 0;
    for (index = 0u; index < tensor->rank; ++index) {
        if (!tensor->dims[index] || elements > ULLONG_MAX / tensor->dims[index]) return 0;
        elements *= tensor->dims[index];
    }
    return elements <= ULLONG_MAX / 2ull && tensor->data_bytes == elements * 2ull;
}

static int qwen_vision_tensor_name(const yvex_qwen3_5_architecture *architecture,
                                   const char *name)
{
    static const char *const block_suffixes[] = {
        "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
        "mlp.linear_fc1.bias", "mlp.linear_fc1.weight", "mlp.linear_fc2.bias",
        "mlp.linear_fc2.weight", "norm1.bias", "norm1.weight", "norm2.bias",
        "norm2.weight"};
    static const char *const global_names[] = {
        "model.visual.merger.linear_fc1.bias", "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc2.bias", "model.visual.merger.linear_fc2.weight",
        "model.visual.merger.norm.bias", "model.visual.merger.norm.weight",
        "model.visual.patch_embed.proj.bias", "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight"};
    unsigned long long layer;
    const char *suffix;
    size_t index;

    if (qwen_indexed_name(name, "model.visual.blocks.", &layer, &suffix)) {
        if (layer >= architecture->vision.depth) return 0;
        for (index = 0u; index < sizeof(block_suffixes) / sizeof(block_suffixes[0]); ++index)
            if (strcmp(suffix, block_suffixes[index]) == 0) return 1;
        return 0;
    }
    for (index = 0u; index < sizeof(global_names) / sizeof(global_names[0]); ++index)
        if (strcmp(name, global_names[index]) == 0) return 1;
    return 0;
}

static int qwen_mtp_tensor_name(const yvex_qwen3_5_architecture *architecture,
                                const char *name)
{
    static const char *const layer_suffixes[] = {
        "input_layernorm.weight", "mlp.down_proj.weight", "mlp.gate_proj.weight",
        "mlp.up_proj.weight", "post_attention_layernorm.weight",
        "self_attn.k_norm.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight"};
    static const char *const global_names[] = {
        "mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight"};
    unsigned long long layer;
    const char *suffix;
    size_t index;

    if (qwen_indexed_name(name, "mtp.layers.", &layer, &suffix)) {
        if (layer >= architecture->text.mtp_hidden_layers) return 0;
        for (index = 0u; index < sizeof(layer_suffixes) / sizeof(layer_suffixes[0]); ++index)
            if (strcmp(suffix, layer_suffixes[index]) == 0) return 1;
        return 0;
    }
    for (index = 0u; index < sizeof(global_names) / sizeof(global_names[0]); ++index)
        if (strcmp(name, global_names[index]) == 0) return 1;
    return 0;
}

static int qwen_tensor_classify(
    const yvex_qwen3_5_architecture *architecture,
    const yvex_native_weight_info *tensor, yvex_qwen3_5_tensor_binding *binding,
    yvex_qwen3_5_failure *failure, yvex_error *err)
{
    const char *suffix = NULL;
    unsigned long long layer = YVEX_QWEN3_5_GLOBAL_TENSOR_LAYER;
    yvex_qwen3_5_tensor_role role = YVEX_QWEN3_5_ROLE_UNKNOWN;

    if (binding) memset(binding, 0, sizeof(*binding));
    if (!architecture || !tensor || !tensor->name || !binding)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_INVALID_ARGUMENT,
                           "tensor", "tensor classification inputs are required",
                           YVEX_ERR_INVALID_ARG, err);
    if (strcmp(tensor->name, "model.language_model.embed_tokens.weight") == 0)
        role = YVEX_QWEN3_5_ROLE_TOKEN_EMBEDDING;
    else if (strcmp(tensor->name, "model.language_model.norm.weight") == 0)
        role = YVEX_QWEN3_5_ROLE_OUTPUT_NORM;
    else if (strcmp(tensor->name, "lm_head.weight") == 0)
        role = YVEX_QWEN3_5_ROLE_OUTPUT_HEAD;
    else if (qwen_indexed_name(tensor->name, "model.language_model.layers.",
                               &layer, &suffix) && layer < architecture->text.layer_count) {
        role = qwen_suffix_role(suffix, qwen_common_layer_tensors,
                                sizeof(qwen_common_layer_tensors) /
                                    sizeof(qwen_common_layer_tensors[0]));
        if (role == YVEX_QWEN3_5_ROLE_UNKNOWN &&
            architecture->text.layers[layer] == YVEX_QWEN3_5_LAYER_FULL_ATTENTION)
            role = qwen_suffix_role(suffix, qwen_full_attention_tensors,
                                    sizeof(qwen_full_attention_tensors) /
                                        sizeof(qwen_full_attention_tensors[0]));
        if (role == YVEX_QWEN3_5_ROLE_UNKNOWN &&
            architecture->text.layers[layer] == YVEX_QWEN3_5_LAYER_LINEAR_ATTENTION)
            role = qwen_suffix_role(suffix, qwen_delta_tensors,
                                    sizeof(qwen_delta_tensors) /
                                        sizeof(qwen_delta_tensors[0]));
    }
    if (role != YVEX_QWEN3_5_ROLE_UNKNOWN) {
        if (!qwen_text_tensor_shape(&architecture->text, tensor, role))
            return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_ROLE,
                               tensor->name, "text tensor dtype or geometry disagrees with the pinned architecture",
                               YVEX_ERR_FORMAT, err);
        binding->classification = YVEX_QWEN3_5_TENSOR_TEXT_EXECUTION_REQUIRED;
        binding->role = role;
        binding->layer_index = layer;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (qwen_vision_tensor_name(architecture, tensor->name)) {
        binding->classification = YVEX_QWEN3_5_TENSOR_VISION_DEFERRED;
        binding->role = YVEX_QWEN3_5_ROLE_VISION_COMPONENT;
    } else if (qwen_mtp_tensor_name(architecture, tensor->name)) {
        binding->classification = YVEX_QWEN3_5_TENSOR_MTP_DEFERRED;
        binding->role = YVEX_QWEN3_5_ROLE_MTP_COMPONENT;
    } else {
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_ROLE,
                           tensor->name, "pinned source tensor has no admitted semantic classification",
                           YVEX_ERR_FORMAT, err);
    }
    if (!qwen_deferred_tensor_valid(tensor))
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_ROLE,
                           tensor->name, "deferred tensor storage geometry is invalid",
                           YVEX_ERR_FORMAT, err);
    binding->layer_index = layer;
    yvex_error_clear(err);
    return YVEX_OK;
}

static unsigned long long qwen_expected_role_count(
    const yvex_qwen3_5_architecture *architecture, yvex_qwen3_5_tensor_role role)
{
    switch (role) {
    case YVEX_QWEN3_5_ROLE_TOKEN_EMBEDDING:
    case YVEX_QWEN3_5_ROLE_OUTPUT_NORM:
    case YVEX_QWEN3_5_ROLE_OUTPUT_HEAD:
        return 1ull;
    case YVEX_QWEN3_5_ROLE_INPUT_NORM:
    case YVEX_QWEN3_5_ROLE_FFN_GATE:
    case YVEX_QWEN3_5_ROLE_FFN_UP:
    case YVEX_QWEN3_5_ROLE_FFN_DOWN:
    case YVEX_QWEN3_5_ROLE_POST_ATTENTION_NORM:
        return architecture->text.layer_count;
    case YVEX_QWEN3_5_ROLE_ATTENTION_Q:
    case YVEX_QWEN3_5_ROLE_ATTENTION_K:
    case YVEX_QWEN3_5_ROLE_ATTENTION_V:
    case YVEX_QWEN3_5_ROLE_ATTENTION_OUT:
    case YVEX_QWEN3_5_ROLE_ATTENTION_Q_NORM:
    case YVEX_QWEN3_5_ROLE_ATTENTION_K_NORM:
        return architecture->text.full_attention_layers;
    case YVEX_QWEN3_5_ROLE_DELTA_DECAY_LOG:
    case YVEX_QWEN3_5_ROLE_DELTA_CONVOLUTION:
    case YVEX_QWEN3_5_ROLE_DELTA_TIME_BIAS:
    case YVEX_QWEN3_5_ROLE_DELTA_DECAY_PROJECTION:
    case YVEX_QWEN3_5_ROLE_DELTA_BETA_PROJECTION:
    case YVEX_QWEN3_5_ROLE_DELTA_QKV_PROJECTION:
    case YVEX_QWEN3_5_ROLE_DELTA_OUTPUT_GATE:
    case YVEX_QWEN3_5_ROLE_DELTA_OUTPUT_NORM:
    case YVEX_QWEN3_5_ROLE_DELTA_OUTPUT:
        return architecture->text.linear_attention_layers;
    case YVEX_QWEN3_5_ROLE_VISION_COMPONENT:
        return architecture->vision.depth * 12ull + 9ull;
    case YVEX_QWEN3_5_ROLE_MTP_COMPONENT:
        return architecture->text.mtp_hidden_layers
                   ? architecture->text.mtp_hidden_layers * 11ull + 4ull : 0ull;
    default:
        return 0ull;
    }
}

static int qwen_tensor_identity_update(yvex_sha256 *hash,
                                       const yvex_native_weight_info *tensor,
                                       const yvex_qwen3_5_tensor_binding *binding)
{
    unsigned int dimension;

    if (!yvex_sha256_update_text(hash, tensor->name) ||
        !yvex_sha256_update_text(hash, tensor->shard_path) ||
        !yvex_sha256_update_u64(hash, (unsigned long long)tensor->dtype) ||
        !yvex_sha256_update_u64(hash, tensor->rank))
        return 0;
    for (dimension = 0u; dimension < tensor->rank; ++dimension)
        if (!yvex_sha256_update_u64(hash, tensor->dims[dimension])) return 0;
    return yvex_sha256_update_u64(hash, tensor->data_start) &&
           yvex_sha256_update_u64(hash, tensor->data_end) &&
           yvex_sha256_update_u64(hash, binding->classification) &&
           yvex_sha256_update_u64(hash, binding->role) &&
           yvex_sha256_update_u64(hash, binding->layer_index);
}

typedef const yvex_native_weight_info *(*qwen_tensor_at_fn)(
    const void *source, unsigned long long index);

static const yvex_native_weight_info *qwen_table_tensor_at(
    const void *source, unsigned long long index)
{
    return yvex_native_weight_table_at(source, index);
}

static const yvex_native_weight_info *qwen_snapshot_tensor_at(
    const void *source, unsigned long long index)
{
    return yvex_source_tensor_snapshot_at(source, index);
}

static int qwen_tensor_rows_audit(
    const yvex_qwen3_5_architecture *architecture,
    const void *source, unsigned long long count, qwen_tensor_at_fn tensor_at,
    yvex_qwen3_5_tensor_inventory *inventory,
    yvex_qwen3_5_failure *failure, yvex_error *err)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long index, expected_text, expected_vision, expected_mtp;
    unsigned int role;

    if (inventory) memset(inventory, 0, sizeof(*inventory));
    if (!architecture || !source || !tensor_at || !inventory)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_INVALID_ARGUMENT,
                           "inventory", "tensor inventory audit inputs are required",
                           YVEX_ERR_INVALID_ARG, err);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.qwen3-5.tensor-role-map.v1") ||
        !yvex_sha256_update_text(&hash, architecture->architecture_identity) ||
        !yvex_sha256_update_u64(&hash, count))
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_INVENTORY,
                           "identity", "tensor role map identity initialization failed",
                           YVEX_ERR_STATE, err);
    for (index = 0ull; index < count; ++index) {
        const yvex_native_weight_info *tensor = tensor_at(source, index);
        yvex_qwen3_5_tensor_binding binding;

        if (!tensor)
            return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_INVENTORY,
                               "tensor", "tensor inventory row disappeared during audit",
                               YVEX_ERR_STATE, err);
        if (qwen_tensor_classify(architecture, tensor, &binding, failure, err) != YVEX_OK)
            return yvex_error_code(err);
        if (inventory->tensor_bytes > ULLONG_MAX - tensor->data_bytes)
            return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_INVENTORY,
                               tensor->name, "tensor inventory byte count overflow",
                               YVEX_ERR_BOUNDS, err);
        inventory->tensor_count++;
        inventory->tensor_bytes += tensor->data_bytes;
        inventory->class_counts[binding.classification]++;
        inventory->class_bytes[binding.classification] += tensor->data_bytes;
        inventory->role_counts[binding.role]++;
        if (!qwen_tensor_identity_update(&hash, tensor, &binding))
            return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_INVENTORY,
                               tensor->name, "tensor role map identity update failed",
                               YVEX_ERR_STATE, err);
    }
    for (role = 1u; role < YVEX_QWEN3_5_ROLE_COUNT; ++role)
        if (inventory->role_counts[role] !=
            qwen_expected_role_count(architecture, (yvex_qwen3_5_tensor_role)role))
            return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_INVENTORY,
                               "role-population", "tensor role population is incomplete or duplicated",
                               YVEX_ERR_FORMAT, err);
    expected_text = 3ull + architecture->text.layer_count * 5ull +
                    architecture->text.full_attention_layers * 6ull +
                    architecture->text.linear_attention_layers * 9ull;
    expected_vision = architecture->vision.depth * 12ull + 9ull;
    expected_mtp = architecture->text.mtp_hidden_layers
                       ? architecture->text.mtp_hidden_layers * 11ull + 4ull : 0ull;
    if (inventory->class_counts[YVEX_QWEN3_5_TENSOR_TEXT_EXECUTION_REQUIRED] !=
            expected_text ||
        inventory->class_counts[YVEX_QWEN3_5_TENSOR_VISION_DEFERRED] !=
            expected_vision ||
        inventory->class_counts[YVEX_QWEN3_5_TENSOR_MTP_DEFERRED] != expected_mtp ||
        inventory->class_counts[YVEX_QWEN3_5_TENSOR_SOURCE_METADATA] != 0ull ||
        inventory->class_counts[YVEX_QWEN3_5_TENSOR_UNKNOWN] != 0ull ||
        inventory->tensor_count != expected_text + expected_vision + expected_mtp ||
        !yvex_sha256_final(&hash, digest))
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_TENSOR_INVENTORY,
                           "population", "pinned tensor inventory population disagrees with the architecture",
                           YVEX_ERR_FORMAT, err);
    yvex_sha256_hex(digest, inventory->role_map_identity);
    inventory->complete = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int qwen_tensor_inventory_audit(
    const yvex_qwen3_5_architecture *architecture,
    const yvex_native_weight_table *weights,
    yvex_qwen3_5_tensor_inventory *inventory,
    yvex_qwen3_5_failure *failure, yvex_error *err)
{
    return qwen_tensor_rows_audit(
        architecture, weights,
        weights ? yvex_native_weight_table_count(weights) : 0ull,
        qwen_table_tensor_at, inventory, failure, err);
}

static int qwen_tensor_snapshot_audit(
    const yvex_qwen3_5_architecture *architecture,
    const yvex_source_tensor_snapshot *snapshot,
    yvex_qwen3_5_tensor_inventory *inventory,
    yvex_qwen3_5_failure *failure, yvex_error *err)
{
    yvex_source_tensor_snapshot_facts facts = {0};

    if (!snapshot ||
        yvex_source_tensor_snapshot_facts_get(snapshot, &facts, err) != YVEX_OK)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_INVALID_ARGUMENT,
                           "snapshot", "retained tensor snapshot facts are required",
                           YVEX_ERR_INVALID_ARG, err);
    return qwen_tensor_rows_audit(
        architecture, snapshot, facts.tensor_count, qwen_snapshot_tensor_at,
        inventory, failure, err);
}

static const char *qwen_tensor_class_name(yvex_qwen3_5_tensor_class classification)
{
    static const char *const names[] = {
        "unknown", "text-execution-required", "vision-deferred",
        "mtp-deferred", "source-metadata"};

    return (unsigned int)classification < sizeof(names) / sizeof(names[0])
               ? names[(unsigned int)classification] : "unknown";
}

static const char *qwen_tensor_role_name(yvex_qwen3_5_tensor_role role)
{
    static const char *const names[] = {
        "unknown", "token-embedding", "output-norm", "output-head", "input-norm",
        "ffn-gate", "ffn-up", "ffn-down", "post-attention-norm", "attention-q",
        "attention-k", "attention-v", "attention-out", "attention-q-norm",
        "attention-k-norm", "delta-decay-log", "delta-convolution", "delta-time-bias",
        "delta-decay-projection", "delta-beta-projection", "delta-qkv-projection",
        "delta-output-gate", "delta-output-norm", "delta-output", "vision-component",
        "mtp-component", "source-metadata"};

    return (unsigned int)role < sizeof(names) / sizeof(names[0])
               ? names[(unsigned int)role] : "unknown";
}

static int qwen_hash_double(yvex_sha256 *hash, double value)
{
    int exponent = 0;
    int negative = signbit(value) != 0;
    double fraction = frexp(fabs(value), &exponent);
    unsigned long long mantissa =
        (unsigned long long)llround(ldexp(fraction, 53));

    return yvex_sha256_update_u64(hash, (unsigned long long)negative) &&
           yvex_sha256_update_u64(hash, (unsigned long long)(exponent + 4096)) &&
           yvex_sha256_update_u64(hash, mantissa);
}

static int qwen_identity(yvex_qwen3_5_architecture *architecture)
{
    const yvex_qwen3_5_text_architecture *text = &architecture->text;
    const yvex_qwen3_5_vision_architecture *vision = &architecture->vision;
    const yvex_qwen3_5_generation_policy *generation = &architecture->generation;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256 hash;
    unsigned long long index;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.model.qwen3_5.text-specialization.v1") ||
        !yvex_sha256_update_text(&hash, architecture->product_id) ||
        !yvex_sha256_update_text(&hash, architecture->source_revision) ||
        !yvex_sha256_update_u64(&hash, text->hidden_size) ||
        !yvex_sha256_update_u64(&hash, text->layer_count) ||
        !yvex_sha256_update_u64(&hash, text->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, text->intermediate_size) ||
        !yvex_sha256_update_u64(&hash, text->maximum_positions) ||
        !yvex_sha256_update_u64(&hash, text->full_attention_interval) ||
        !yvex_sha256_update_u64(&hash, text->attention_heads) ||
        !yvex_sha256_update_u64(&hash, text->kv_heads) ||
        !yvex_sha256_update_u64(&hash, text->attention_head_dimension) ||
        !yvex_sha256_update_u64(&hash, text->rotary_dimension) ||
        !yvex_sha256_update_u64(&hash, text->rope_theta) ||
        !yvex_sha256_update_u64(&hash, text->linear_key_heads) ||
        !yvex_sha256_update_u64(&hash, text->linear_value_heads) ||
        !yvex_sha256_update_u64(&hash, text->linear_key_head_dimension) ||
        !yvex_sha256_update_u64(&hash, text->linear_value_head_dimension) ||
        !yvex_sha256_update_u64(&hash, text->linear_convolution_kernel) ||
        !yvex_sha256_update_u64(&hash, text->mtp_hidden_layers) ||
        !qwen_hash_double(&hash, text->rms_norm_epsilon) ||
        !qwen_hash_double(&hash, text->partial_rotary_factor))
        return 0;
    for (index = 0ull; index < text->layer_count; ++index)
        if (!yvex_sha256_update_u64(&hash, (unsigned long long)text->layers[index])) return 0;
    if (!yvex_sha256_update_u64(&hash, vision->depth) ||
        !yvex_sha256_update_u64(&hash, vision->hidden_size) ||
        !yvex_sha256_update_u64(&hash, vision->output_hidden_size) ||
        !yvex_sha256_update_u64(&hash, generation->stop_token_count))
        return 0;
    for (index = 0ull; index < generation->stop_token_count; ++index)
        if (!yvex_sha256_update_u64(&hash, generation->stop_token_ids[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, architecture->architecture_identity);
    return 1;
}

static int qwen_model_open(yvex_qwen3_5_model **out,
                           const yvex_source_verification *verification,
                           yvex_qwen3_5_failure *failure, yvex_error *err)
{
    yvex_qwen3_5_model *model = NULL;
    char config_path[YVEX_PATH_CAP], generation_path[YVEX_PATH_CAP];
    char *config = NULL, *generation = NULL;
    size_t config_length = 0u, generation_length = 0u;
    int rc = YVEX_OK;

    if (out) *out = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    if (!out || !verification)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_INVALID_ARGUMENT,
                           "verification", "source verification and output are required",
                           YVEX_ERR_INVALID_ARG, err);
    if (!verification->verified || !verification->config_valid)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_SOURCE_NOT_VERIFIED,
                           "verification", "exact source verification is required",
                           YVEX_ERR_STATE, err);
    if (strcmp(verification->repository_id,
               YVEX_SOURCE_QWEN3_8_27B_REPOSITORY) != 0 ||
        strcmp(verification->revision,
               YVEX_SOURCE_QWEN3_8_27B_REVISION) != 0 ||
        strcmp(verification->model_type,
               YVEX_SOURCE_QWEN3_8_27B_CONFIG_TYPE) != 0 ||
        strcmp(verification->architecture,
               YVEX_SOURCE_QWEN3_8_27B_CONFIG_ARCHITECTURE) != 0)
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_SOURCE_IDENTITY,
                           "source", "source identity is not the pinned Qwen3.8-27B release",
                           YVEX_ERR_FORMAT, err);
    if (!yvex_source_path_join(config_path, sizeof(config_path),
                               verification->resolved_source_path, "config.json") ||
        !yvex_source_path_join(generation_path, sizeof(generation_path),
                               verification->resolved_source_path,
                               "generation_config.json"))
        return qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_MISSING_CONFIG,
                           "path", "source configuration path exceeds bounds",
                           YVEX_ERR_BOUNDS, err);
    config = yvex_read_bounded_file(config_path, QWEN_CONFIG_CAP,
                                    &config_length, err);
    generation = yvex_read_bounded_file(generation_path, QWEN_CONFIG_CAP,
                                        &generation_length, err);
    if (!config || !generation) {
        rc = qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_MISSING_CONFIG,
                         !config ? "config.json" : "generation_config.json",
                         "pinned source configuration is unavailable",
                         yvex_error_code(err) == YVEX_ERR_NOMEM
                             ? YVEX_ERR_NOMEM : YVEX_ERR_IO,
                         err);
        goto cleanup;
    }
    model = calloc(1u, sizeof(*model));
    if (!model) {
        rc = qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_ALLOCATION,
                         "model", "Qwen semantic model allocation failed",
                         YVEX_ERR_NOMEM, err);
        goto cleanup;
    }
    yvex_core_text_copy(model->architecture.product_id,
                        sizeof(model->architecture.product_id),
                        YVEX_QWEN3_8_27B_TARGET_ID);
    yvex_core_text_copy(model->architecture.semantic_family,
                        sizeof(model->architecture.semantic_family),
                        YVEX_QWEN3_5_FAMILY_KEY);
    yvex_core_text_copy(model->architecture.source_revision,
                        sizeof(model->architecture.source_revision),
                        verification->revision);
    if (!qwen_parse_config(config, config_length, &model->architecture)) {
        rc = qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_MALFORMED_CONFIG,
                         "config.json", "Qwen source configuration is malformed or incomplete",
                         YVEX_ERR_FORMAT, err);
        goto cleanup;
    }
    if (!qwen_parse_generation(generation, generation_length,
                               &model->architecture.generation)) {
        rc = qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_GENERATION_POLICY,
                         "generation_config.json", "Qwen generation policy is malformed or incomplete",
                         YVEX_ERR_FORMAT, err);
        goto cleanup;
    }
    rc = qwen_validate(&model->architecture, failure, err);
    if (rc == YVEX_OK && !qwen_identity(&model->architecture))
        rc = qwen_refuse(failure, YVEX_QWEN3_5_FAILURE_CONFIGURATION,
                         "identity", "Qwen semantic identity derivation failed",
                         YVEX_ERR_STATE, err);
    if (rc == YVEX_OK) {
        *out = model;
        model = NULL;
        yvex_error_clear(err);
    }
cleanup:
    free(config);
    free(generation);
    free(model);
    return rc;
}

static void qwen_model_close(yvex_qwen3_5_model **model)
{
    if (!model) return;
    free(*model);
    *model = NULL;
}

static const yvex_qwen3_5_architecture *qwen_architecture(
    const yvex_qwen3_5_model *model)
{
    return model ? &model->architecture : NULL;
}

static yvex_qwen3_5_layer_kind qwen_layer_kind(
    const yvex_qwen3_5_model *model, unsigned long long layer)
{
    return model && layer < model->architecture.text.layer_count
               ? model->architecture.text.layers[layer]
               : 0;
}

static const char *qwen_failure_name(yvex_qwen3_5_failure_code code)
{
    static const char *const names[] = {
        "none", "invalid-argument", "source-not-verified", "source-identity",
        "missing-config", "malformed-config", "configuration",
        "generation-policy", "tensor-role", "tensor-inventory", "allocation"};

    return (unsigned int)code < sizeof(names) / sizeof(names[0])
               ? names[(unsigned int)code]
               : "unknown";
}

static const char qwen_reasoning_xhigh[] =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
    "and clarity in the final answer.";
static const char qwen_reasoning_low[] =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";
static const char qwen_tools_prefix[] =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const char qwen_tools_suffix[] =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>"
    "\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the "
    "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is no "
    "function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>";

static const yvex_conversation_protocol qwen_conversation = {
    .schema_version = YVEX_CONVERSATION_PROTOCOL_SCHEMA_V2,
    .family_adapter_id = YVEX_QWEN3_5_ADAPTER_ID,
    .family_adapter_version = YVEX_QWEN3_5_ADAPTER_VERSION,
    .architecture = YVEX_QWEN3_5_FAMILY_KEY,
    .source_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    .source_encoding_path = "tokenizer_config.json#chat_template",
    .source_encoding_identity =
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    .bos = "", .eos = "<|im_end|>",
    .system = "<|im_start|>system\n",
    .user = "<|im_start|>user\n",
    .assistant = "<|im_start|>assistant\n",
    .message_end = "<|im_end|>\n",
    .latest_reminder = "",
    .thinking_start = "<think>", .thinking_start_suffix = "\n",
    .thinking_end_prefix = "\n", .thinking_end = "</think>",
    .thinking_end_suffix = "\n\n",
    .tool_result_start = "\n<tool_response>\n",
    .tool_result_end = "\n</tool_response>",
    .tool_result_group_start = "<|im_start|>user",
    .dsml = "<tool_call>", .tool_calls_start = "",
    .tool_calls_end = "", .tool_invoke_start = "<tool_call>\n<function=",
    .tool_invoke_name_end = ">\n",
    .tool_invoke_end = "</function>\n</tool_call>",
    .tool_parameter_start = "<parameter=",
    .tool_parameter_name_end = ">\n", .tool_parameter_kind_end = "",
    .tool_parameter_end = "\n</parameter>\n",
    .reasoning_effort_low = qwen_reasoning_low,
    .reasoning_effort_max = qwen_reasoning_xhigh,
    .tools_prefix = qwen_tools_prefix, .tools_suffix = qwen_tools_suffix,
    .response_format_prefix = "",
    .grammar = YVEX_CONVERSATION_GRAMMAR_ROLE_ENVELOPED,
    .tool_grammar = YVEX_CONVERSATION_TOOL_GRAMMAR_XML_ELEMENTS,
    .default_reasoning_policy = YVEX_REASONING_MAXIMUM,
    .drop_prior_reasoning_by_default = 0, .tools_preserve_reasoning = 1,
    .tool_results_merge_into_user = 1,
    .tokenizer_model = "gpt2", .tokenizer_pre = "qwen2",
    .tokenizer_json_identity =
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    .tokenizer_config_identity =
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    .vocabulary_size = 248077ull, .base_vocabulary_size = 248044ull,
    .merge_count = 247587ull, .added_token_count = 33ull,
    .special_token_count = 21ull,
    .eos_token_id = 248046u, .pad_token_id = 248044u,
    .eos_present = 1, .pad_present = 1};

const yvex_conversation_protocol *yvex_model_qwen3_5_conversation(void)
{
    return &qwen_conversation;
}

const yvex_qwen3_5_api *yvex_model_register_qwen3_5(void)
{
    static const yvex_qwen3_5_api api = {
        .schema_version = 3u,
        .open = qwen_model_open,
        .close = qwen_model_close,
        .architecture = qwen_architecture,
        .layer_kind = qwen_layer_kind,
        .tensor_classify = qwen_tensor_classify,
        .tensor_inventory_audit = qwen_tensor_inventory_audit,
        .tensor_snapshot_audit = qwen_tensor_snapshot_audit,
        .failure_name = qwen_failure_name,
        .tensor_class_name = qwen_tensor_class_name,
        .tensor_role_name = qwen_tensor_role_name};

    return &api;
}
