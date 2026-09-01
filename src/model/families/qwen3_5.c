/* Parse and seal Qwen3.5-family semantics from one authenticated source revision. */
#include <yvex/internal/families/qwen3_5.h>

#include <yvex/internal/core.h>

#include <errno.h>
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
    if (strcmp(verification->repository_id, "Qwen/Qwen3.8-27B") != 0 ||
        strcmp(verification->revision,
               "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0") != 0 ||
        strcmp(verification->model_type, "qwen3_5") != 0 ||
        strcmp(verification->architecture,
               "Qwen3_5ForConditionalGeneration") != 0)
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
        "generation-policy", "allocation"};

    return (unsigned int)code < sizeof(names) / sizeof(names[0])
               ? names[(unsigned int)code]
               : "unknown";
}

const yvex_qwen3_5_api *yvex_model_register_qwen3_5(void)
{
    static const yvex_qwen3_5_api api = {
        .schema_version = 1u,
        .open = qwen_model_open,
        .close = qwen_model_close,
        .architecture = qwen_architecture,
        .layer_kind = qwen_layer_kind,
        .failure_name = qwen_failure_name};

    return &api;
}
