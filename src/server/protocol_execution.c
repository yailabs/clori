/* Encode and validate the fixed execution-truth subrecords carried by protocol v20. */
#include "src/server/private.h"

#include <math.h>
#include <string.h>

static void truth_put_u64(unsigned char *out, unsigned long long value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        out[index] = (unsigned char)(value >> (56u - index * 8u));
}

static unsigned long long truth_get_u64(const unsigned char *in)
{
    unsigned long long value = 0ull;
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        value = (value << 8u) | in[index];
    return value;
}

static void truth_put_double(unsigned char *out, double value)
{
    unsigned long long bits;
    memcpy(&bits, &value, sizeof(bits));
    truth_put_u64(out, bits);
}

static double truth_get_double(const unsigned char *in)
{
    unsigned long long bits = truth_get_u64(in);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int bool_valid(int value)
{
    return value == 0 || value == 1;
}

int yvex_server_execution_capacity_valid(
    const yvex_execution_capacity_summary *value)
{
    return value &&
           value->schema_version == YVEX_EXECUTION_CAPACITY_SCHEMA_V1 &&
           value->session_capacity && value->runnable_work_capacity &&
           value->physical_sequence_width &&
           value->runnable_work_capacity <= value->session_capacity &&
           bool_valid(value->cooperative_scheduling_ready) &&
           bool_valid(value->compatible_operation_batching_ready) &&
           bool_valid(value->continuous_batching_ready) &&
           (!value->cooperative_scheduling_ready ||
            value->runnable_work_capacity > 1ull) &&
           (!value->compatible_operation_batching_ready ||
            value->physical_sequence_width > 1ull) &&
           (!value->continuous_batching_ready ||
            value->physical_sequence_width > 1ull);
}

int yvex_server_execution_measurement_valid(
    const yvex_execution_measurement *value)
{
    const unsigned long long allowed =
        YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE |
        YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE |
        YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE |
        YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE;
    if (!value) return 0;
    if (!value->schema_version) {
        yvex_execution_measurement empty = {0};
        return memcmp(value, &empty, sizeof(empty)) == 0;
    }
    if (value->schema_version != YVEX_EXECUTION_MEASUREMENT_SCHEMA_V1 ||
        value->scope == YVEX_EXECUTION_SCOPE_UNAVAILABLE ||
        value->scope > YVEX_EXECUTION_SCOPE_UNATTRIBUTED ||
        value->clock > YVEX_EXECUTION_CLOCK_MIXED ||
        value->composition == YVEX_EXECUTION_COMPOSITION_UNAVAILABLE ||
        value->composition > YVEX_EXECUTION_COMPOSITION_OVERLAPPING ||
        value->work_unit > YVEX_EXECUTION_WORK_OPERATIONS ||
        (value->available & ~allowed) || !isfinite(value->cumulative_rate) ||
        !isfinite(value->rolling_rate))
        return 0;
    if (!(value->available & YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE) &&
        value->duration_ns)
        return 0;
    if ((value->available & YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE) &&
        (!value->duration_ns ||
         value->clock == YVEX_EXECUTION_CLOCK_UNAVAILABLE))
        return 0;
    if (!(value->available & YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE) &&
        value->total_units)
        return 0;
    if ((value->available &
         YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE) &&
        !value->total_units)
        return 0;
    if ((value->available & YVEX_EXECUTION_MEASUREMENT_DENOMINATOR_AVAILABLE) &&
        value->completed_units > value->total_units)
        return 0;
    if (!(value->available & YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE) &&
        value->cumulative_rate != 0.0)
        return 0;
    if ((value->available &
         YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE) &&
        (!value->completed_units || !value->duration_ns ||
         value->cumulative_rate <= 0.0))
        return 0;
    if (!(value->available & YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE) &&
        (value->rolling_rate != 0.0 || value->rolling_units ||
         value->rolling_duration_ns || value->rolling_window_units))
        return 0;
    if ((value->available &
         YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE) &&
        (!value->rolling_units || !value->rolling_duration_ns ||
         !value->rolling_window_units ||
         value->rolling_units > value->rolling_window_units ||
         value->rolling_units > value->completed_units ||
         ((value->available &
           YVEX_EXECUTION_MEASUREMENT_DURATION_AVAILABLE) &&
          value->rolling_duration_ns > value->duration_ns) ||
         value->rolling_rate <= 0.0))
        return 0;
    if ((value->completed_units || value->total_units || value->duration_ns) &&
        value->work_unit == YVEX_EXECUTION_WORK_NONE)
        return 0;
    return !(value->available &
             (YVEX_EXECUTION_MEASUREMENT_CUMULATIVE_RATE_AVAILABLE |
              YVEX_EXECUTION_MEASUREMENT_ROLLING_RATE_AVAILABLE)) ||
           value->work_unit != YVEX_EXECUTION_WORK_NONE;
}

int yvex_server_execution_resource_valid(
    const yvex_execution_resource_summary *value)
{
    const unsigned long long allowed =
        YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_ARENA_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_TRANSIENT_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE |
        YVEX_EXECUTION_RESOURCE_UNIFIED_MEMORY;
    if (!value) return 0;
    if (!value->schema_version) {
        yvex_execution_resource_summary empty = {0};
        return memcmp(value, &empty, sizeof(empty)) == 0;
    }
    if (value->schema_version != YVEX_EXECUTION_RESOURCE_SCHEMA_V1 ||
        value->placement > YVEX_EXECUTION_PLACEMENT_COMPOSITE ||
        (value->available & ~allowed))
        return 0;
    if ((value->available & YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE) &&
        !value->component_count)
        return 0;
    if ((value->available &
         YVEX_EXECUTION_RESOURCE_PHYSICAL_RESIDENCY_AVAILABLE) &&
        (value->placement == YVEX_EXECUTION_PLACEMENT_MANAGED_UNIFIED ||
         value->placement ==
             YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED_DEVICE_ADDRESSABLE))
        return 0;
    if ((value->available & YVEX_EXECUTION_RESOURCE_UNIFIED_MEMORY) &&
        value->placement != YVEX_EXECUTION_PLACEMENT_MANAGED_UNIFIED &&
        value->placement !=
            YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED_DEVICE_ADDRESSABLE &&
        value->placement != YVEX_EXECUTION_PLACEMENT_COMPOSITE)
        return 0;
    if ((value->placement == YVEX_EXECUTION_PLACEMENT_MANAGED_UNIFIED ||
         value->placement ==
             YVEX_EXECUTION_PLACEMENT_ARTIFACT_MAPPED_DEVICE_ADDRESSABLE) &&
        value->model_explicit_device_bytes)
        return 0;
    if (!(value->available & YVEX_EXECUTION_RESOURCE_MODEL_AVAILABLE) &&
        (value->component_count || value->model_artifact_bytes ||
         value->model_mapped_bytes || value->model_prepared_bytes ||
         value->model_explicit_host_bytes || value->model_explicit_device_bytes ||
         value->model_device_addressable_bytes))
        return 0;
    if (!(value->available & YVEX_EXECUTION_RESOURCE_SESSION_AVAILABLE) &&
        (value->session_attention_allocated_bytes ||
         value->session_attention_resident_bytes ||
         value->session_attention_virtual_bytes ||
         value->session_attention_page_table_bytes ||
         value->session_recurrent_state_bytes ||
         value->session_convolution_state_bytes ||
         value->session_candidate_state_bytes ||
         value->session_physical_state_bytes))
        return 0;
    if (value->session_attention_resident_bytes >
            value->session_attention_allocated_bytes ||
        value->session_attention_allocated_bytes >
            value->session_physical_state_bytes ||
        value->session_recurrent_state_bytes >
            value->session_physical_state_bytes ||
        value->session_convolution_state_bytes >
            value->session_physical_state_bytes ||
        value->session_candidate_state_bytes >
            value->session_physical_state_bytes)
        return 0;
    if (!(value->available & YVEX_EXECUTION_RESOURCE_ARENA_AVAILABLE) &&
        (value->activation_arena_current_bytes ||
         value->activation_arena_peak_bytes))
        return 0;
    if (!(value->available & YVEX_EXECUTION_RESOURCE_WORKSPACE_AVAILABLE) &&
        (value->workspace_current_bytes || value->workspace_peak_bytes))
        return 0;
    if (!(value->available & YVEX_EXECUTION_RESOURCE_TRANSIENT_AVAILABLE) &&
        (value->transient_current_bytes || value->transient_peak_bytes))
        return 0;
    if (value->activation_arena_current_bytes >
            value->activation_arena_peak_bytes ||
        value->workspace_current_bytes > value->workspace_peak_bytes ||
        value->transient_current_bytes > value->transient_peak_bytes ||
        value->process_rss_current_bytes > value->process_rss_peak_bytes)
        return 0;
    return (value->available & YVEX_EXECUTION_RESOURCE_PROCESS_AVAILABLE) ||
           (!value->process_rss_current_bytes && !value->process_rss_peak_bytes);
}

int yvex_server_protocol_capacity_encode(
    const yvex_execution_capacity_summary *value,
    unsigned char output[YVEX_SERVER_PROTOCOL_CAPACITY_BYTES])
{
    const unsigned long long values[] = {
        value ? value->schema_version : 0u,
        value ? value->session_capacity : 0ull,
        value ? value->runnable_work_capacity : 0ull,
        value ? value->physical_sequence_width : 0ull,
        value ? ((value->cooperative_scheduling_ready ? 1ull : 0ull) |
                 (value->compatible_operation_batching_ready ? 2ull : 0ull) |
                 (value->continuous_batching_ready ? 4ull : 0ull)) : 0ull};
    unsigned int index;
    if (!output || !yvex_server_execution_capacity_valid(value)) return 0;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        truth_put_u64(output + index * 8u, values[index]);
    return 1;
}

int yvex_server_protocol_capacity_decode(
    const unsigned char *input, unsigned long long count,
    yvex_execution_capacity_summary *value)
{
    unsigned long long flags;
    if (!input || !value || count != YVEX_SERVER_PROTOCOL_CAPACITY_BYTES)
        return 0;
    memset(value, 0, sizeof(*value));
    value->schema_version = (unsigned int)truth_get_u64(input);
    value->session_capacity = truth_get_u64(input + 8u);
    value->runnable_work_capacity = truth_get_u64(input + 16u);
    value->physical_sequence_width = truth_get_u64(input + 24u);
    flags = truth_get_u64(input + 32u);
    if (flags & ~7ull) return 0;
    value->cooperative_scheduling_ready = (flags & 1ull) != 0ull;
    value->compatible_operation_batching_ready = (flags & 2ull) != 0ull;
    value->continuous_batching_ready = (flags & 4ull) != 0ull;
    return yvex_server_execution_capacity_valid(value);
}

int yvex_server_protocol_measurement_encode(
    const yvex_execution_measurement *value,
    unsigned char output[YVEX_SERVER_PROTOCOL_MEASUREMENT_BYTES])
{
    const unsigned long long values[] = {
        value ? value->schema_version : 0u, value ? value->scope : 0u,
        value ? value->clock : 0u, value ? value->composition : 0u,
        value ? value->work_unit : 0u, value ? value->available : 0ull,
        value ? value->completed_units : 0ull,
        value ? value->total_units : 0ull, value ? value->duration_ns : 0ull,
        value ? value->rolling_units : 0ull,
        value ? value->rolling_duration_ns : 0ull,
        value ? value->rolling_window_units : 0ull};
    unsigned int index;
    if (!output || !yvex_server_execution_measurement_valid(value)) return 0;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        truth_put_u64(output + index * 8u, values[index]);
    truth_put_double(output + 96u, value->cumulative_rate);
    truth_put_double(output + 104u, value->rolling_rate);
    return 1;
}

int yvex_server_protocol_measurement_decode(
    const unsigned char *input, unsigned long long count,
    yvex_execution_measurement *value)
{
    if (!input || !value || count != YVEX_SERVER_PROTOCOL_MEASUREMENT_BYTES)
        return 0;
    memset(value, 0, sizeof(*value));
    value->schema_version = (unsigned int)truth_get_u64(input);
    value->scope = (yvex_execution_measurement_scope)truth_get_u64(input + 8u);
    value->clock = (yvex_execution_measurement_clock)truth_get_u64(input + 16u);
    value->composition = (yvex_execution_measurement_composition)truth_get_u64(input + 24u);
    value->work_unit = (yvex_execution_work_unit)truth_get_u64(input + 32u);
    value->available = truth_get_u64(input + 40u);
    value->completed_units = truth_get_u64(input + 48u);
    value->total_units = truth_get_u64(input + 56u);
    value->duration_ns = truth_get_u64(input + 64u);
    value->rolling_units = truth_get_u64(input + 72u);
    value->rolling_duration_ns = truth_get_u64(input + 80u);
    value->rolling_window_units = truth_get_u64(input + 88u);
    value->cumulative_rate = truth_get_double(input + 96u);
    value->rolling_rate = truth_get_double(input + 104u);
    return yvex_server_execution_measurement_valid(value);
}

int yvex_server_protocol_resource_encode(
    const yvex_execution_resource_summary *value,
    unsigned char output[YVEX_SERVER_PROTOCOL_RESOURCE_BYTES])
{
    const unsigned long long values[] = {
        value ? value->schema_version : 0u, value ? value->placement : 0u,
        value ? value->available : 0ull, value ? value->component_count : 0ull,
        value ? value->model_artifact_bytes : 0ull,
        value ? value->model_mapped_bytes : 0ull,
        value ? value->model_prepared_bytes : 0ull,
        value ? value->model_explicit_host_bytes : 0ull,
        value ? value->model_explicit_device_bytes : 0ull,
        value ? value->model_device_addressable_bytes : 0ull,
        value ? value->session_attention_allocated_bytes : 0ull,
        value ? value->session_attention_resident_bytes : 0ull,
        value ? value->session_attention_virtual_bytes : 0ull,
        value ? value->session_attention_page_table_bytes : 0ull,
        value ? value->session_recurrent_state_bytes : 0ull,
        value ? value->session_convolution_state_bytes : 0ull,
        value ? value->session_candidate_state_bytes : 0ull,
        value ? value->session_physical_state_bytes : 0ull,
        value ? value->activation_arena_current_bytes : 0ull,
        value ? value->activation_arena_peak_bytes : 0ull,
        value ? value->workspace_current_bytes : 0ull,
        value ? value->workspace_peak_bytes : 0ull,
        value ? value->transient_current_bytes : 0ull,
        value ? value->transient_peak_bytes : 0ull,
        value ? value->process_rss_current_bytes : 0ull,
        value ? value->process_rss_peak_bytes : 0ull,
        value ? value->logical_upload_bytes : 0ull,
        value ? value->logical_download_bytes : 0ull};
    unsigned int index;
    if (!output || !yvex_server_execution_resource_valid(value)) return 0;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        truth_put_u64(output + index * 8u, values[index]);
    return 1;
}

int yvex_server_protocol_resource_decode(
    const unsigned char *input, unsigned long long count,
    yvex_execution_resource_summary *value)
{
    unsigned int index;
    unsigned long long *fields[26];
    if (!input || !value || count != YVEX_SERVER_PROTOCOL_RESOURCE_BYTES)
        return 0;
    memset(value, 0, sizeof(*value));
    {
    unsigned long long *const bound[] = {
        &value->available, &value->component_count, &value->model_artifact_bytes,
        &value->model_mapped_bytes, &value->model_prepared_bytes,
        &value->model_explicit_host_bytes, &value->model_explicit_device_bytes,
        &value->model_device_addressable_bytes,
        &value->session_attention_allocated_bytes,
        &value->session_attention_resident_bytes,
        &value->session_attention_virtual_bytes,
        &value->session_attention_page_table_bytes,
        &value->session_recurrent_state_bytes,
        &value->session_convolution_state_bytes,
        &value->session_candidate_state_bytes,
        &value->session_physical_state_bytes,
        &value->activation_arena_current_bytes,
        &value->activation_arena_peak_bytes, &value->workspace_current_bytes,
        &value->workspace_peak_bytes, &value->transient_current_bytes,
        &value->transient_peak_bytes, &value->process_rss_current_bytes,
        &value->process_rss_peak_bytes, &value->logical_upload_bytes,
        &value->logical_download_bytes};
    memcpy(fields, bound, sizeof(fields));
    }
    value->schema_version = (unsigned int)truth_get_u64(input);
    value->placement = (yvex_execution_resource_placement)truth_get_u64(input + 8u);
    for (index = 0u; index < sizeof(fields) / sizeof(fields[0]); ++index)
        *fields[index] = truth_get_u64(input + 16u + index * 8u);
    return yvex_server_execution_resource_valid(value);
}
