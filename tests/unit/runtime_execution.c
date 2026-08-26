#include "tests/test.h"

#include <limits.h>
#include <math.h>
#include <string.h>

#include <yvex/internal/backend.h>
#include <yvex/internal/execution.h>

static void execution_test_identity(char output[YVEX_SHA256_HEX_CAP], char digit)
{
    memset(output, digit, YVEX_SHA256_HEX_CAP - 1u);
    output[YVEX_SHA256_HEX_CAP - 1u] = '\0';
}

static int execution_test_planning(void)
{
    char logical[YVEX_SHA256_HEX_CAP], source[YVEX_SHA256_HEX_CAP];
    char schedule[YVEX_SHA256_HEX_CAP], state[YVEX_SHA256_HEX_CAP];
    yvex_model_execution_descriptor_request model_request = {0};
    yvex_model_execution_descriptor model, changed;
    yvex_execution_hardware_profile hardware = {0};
    yvex_execution_workload_profile workload = {0}, independently_scheduled;
    yvex_execution_capacity_plan_request capacity_request = {0};
    yvex_execution_capacity_plan capacity, repeated;
    yvex_execution_state_class_request states[YVEX_MODEL_STATE_CLASS_COUNT] = {{0}};
    unsigned char wire[YVEX_MODEL_EXECUTION_WIRE_BYTES];
    unsigned long long index;
    yvex_error err;

    execution_test_identity(logical, 'a');
    execution_test_identity(source, 'b');
    execution_test_identity(schedule, 'c');
    execution_test_identity(state, 'd');
    model_request.schema_version = YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1;
    model_request.logical_model_identity = logical;
    model_request.source_model_identity = source;
    model_request.attention_schedule_identity = schedule;
    model_request.persistent_state_identity = state;
    model_request.maximum_context = 1048576ull;
    model_request.original_context = 65536ull;
    model_request.rope_scaling = YVEX_MODEL_ROPE_SCALING_YARN;
    model_request.rope_theta = 10000ull;
    model_request.compressed_rope_theta = 160000ull;
    model_request.rope_scaling_factor = 16ull;
    model_request.rope_beta_fast = 32ull;
    model_request.rope_beta_slow = 1ull;
    model_request.layer_count = 43ull;
    model_request.hidden_width = 4096ull;
    model_request.vocabulary_size = 129280ull;
    model_request.attention_heads = 32ull;
    model_request.kv_heads = 32ull;
    model_request.head_width = 128ull;
    model_request.swa_layers = 10ull;
    model_request.csa_layers = 20ull;
    model_request.hca_layers = 13ull;
    model_request.sliding_window = 128ull;
    model_request.minimum_compression_ratio = 4ull;
    model_request.maximum_compression_ratio = 128ull;
    model_request.index_heads = 64ull;
    model_request.index_head_width = 128ull;
    model_request.index_topk = 512ull;
    model_request.residual_streams = 4ull;
    model_request.mhc_sinkhorn_iterations = 20ull;
    model_request.mhc_epsilon = 1e-6;
    model_request.normalization_epsilon = 1e-6;
    model_request.routed_experts = 256ull;
    model_request.experts_per_row = 6ull;
    model_request.shared_experts = 1ull;
    model_request.routed_ffn_width = 2048ull;
    model_request.shared_ffn_width = 2048ull;
    model_request.hash_router_layer_count = 3ull;
    model_request.routed_scaling_factor = 1.5;
    model_request.activation_limit = 10.0;
    model_request.output_input_width = 4096ull;
    model_request.output_vocabulary_size = 129280ull;
    model_request.proposal_width = 5ull;
    model_request.verification_width_maximum = 6ull;
    model_request.draft_layer_count = 3ull;
    model_request.target_feature_count = 3ull;
    model_request.target_feature_layers[0] = 40ull;
    model_request.target_feature_layers[1] = 41ull;
    model_request.target_feature_layers[2] = 42ull;
    model_request.target_feature_width = 4096ull;
    model_request.markov_rank = 256ull;
    model_request.confidence_width = 1ull;
    model_request.persistent_state_class_mask = 0x3ffull;
    model_request.bos_token_id = 0ull;
    model_request.eos_token_id = 1ull;
    model_request.draft_noise_token_id = 128799ull;
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_seal(
                         &model_request, &model, &err) == YVEX_OK,
                     "source-derived model execution descriptor should seal");
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_seal(
                         &model_request, &changed, &err) == YVEX_OK &&
                         strcmp(model.identity, changed.identity) == 0,
                     "model execution descriptor identity should be deterministic");
    model_request.routed_experts = 128ull;
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_seal(
                         &model_request, &changed, &err) == YVEX_OK &&
                         strcmp(model.identity, changed.identity) != 0,
                     "changed synthetic expert geometry should change identity");
    model_request.routed_experts = 256ull;
    model_request.layer_count = 42ull;
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_seal(
                         &model_request, &changed, &err) == YVEX_ERR_INVALID_ARG,
                     "schedule count must refuse a changed synthetic layer count");
    model_request.layer_count = 43ull;
    model_request.maximum_context = 524288ull;
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_seal(
                         &model_request, &changed, &err) == YVEX_OK &&
                         strcmp(model.identity, changed.identity) != 0,
                     "changed synthetic context must change execution identity");
    model_request.maximum_context = 1048576ull;
    model_request.proposal_width = 4ull;
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_seal(
                         &model_request, &changed, &err) == YVEX_OK &&
                         strcmp(model.identity, changed.identity) != 0,
                     "changed synthetic proposal geometry must change execution identity");
    model_request.proposal_width = 5ull;
    execution_test_identity(schedule, 'e');
    model_request.attention_schedule_identity = schedule;
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_seal(
                         &model_request, &changed, &err) == YVEX_OK &&
                         strcmp(model.identity, changed.identity) != 0,
                     "changed synthetic attention schedule must change execution identity");
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_encode(&model, wire, &err) == YVEX_OK &&
                         yvex_model_execution_descriptor_decode(
                             wire, sizeof(wire), &changed, &err) == YVEX_OK &&
                         strcmp(model.identity, changed.identity) == 0,
                     "model execution wire record roundtrips canonically");
    memset(wire + 24u, 0xff, 8u);
    YVEX_TEST_ASSERT(yvex_model_execution_descriptor_decode(
                         wire, sizeof(wire), &changed, &err) == YVEX_ERR_FORMAT,
                     "model execution wire record refuses an invalid enum value");

    hardware.schema_version = YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1;
    hardware.backend = YVEX_BACKEND_KIND_CUDA;
    hardware.admitted_fact_mask =
        (1ull << YVEX_EXECUTION_HARDWARE_FACT_COUNT) - 1ull;
    hardware.compute_major = 12;
    hardware.compute_minor = 1;
    hardware.sm_count = 48ull;
    hardware.copy_engine_count = 2ull;
    hardware.l2_bytes = 24ull * 1024ull * 1024ull;
    hardware.total_memory_bytes = 128ull * 1024ull * 1024ull * 1024ull;
    hardware.usable_memory_bytes = 119ull * 1024ull * 1024ull * 1024ull;
    hardware.sustainable_read_bytes_per_second = 215ull * 1000ull * 1000ull * 1000ull;
    hardware.sustainable_copy_bytes_per_second = 205ull * 1000ull * 1000ull * 1000ull;
    hardware.host_page_bytes = 4096ull;
    hardware.device_page_bytes = 65536ull;
    hardware.unified_addressing = 1;
    hardware.coherent_host_memory = 1;
    hardware.virtual_memory = 1;
    hardware.graph_capture = 1;
    hardware.native_architecture_code = 1;
    memcpy(hardware.name, "gb10-sm121-measured", sizeof("gb10-sm121-measured"));
    YVEX_TEST_ASSERT(yvex_execution_hardware_profile_seal(&hardware, &err) == YVEX_OK,
                     "measured hardware profile should seal");

    workload.schema_version = YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1;
    workload.kind = YVEX_EXECUTION_WORKLOAD_BALANCED_SERVING;
    workload.minimum_session_context = 65536ull;
    workload.requested_session_context = 131072ull;
    workload.concurrent_sequences = 4ull;
    workload.logical_batch_tokens = 2048ull;
    workload.prefill_chunk_tokens = 512ull;
    workload.attention_microbatch_rows = 32ull;
    workload.moe_row_tile = 128ull;
    workload.output_head_rows = 32ull;
    workload.prefix_cache_bytes = 1ull * 1024ull * 1024ull * 1024ull;
    workload.persistent_state_bytes = 1ull * 1024ull * 1024ull * 1024ull;
    workload.system_reserve_bytes = 12ull * 1024ull * 1024ull * 1024ull;
    workload.continuous_batching = 1;
    workload.prefix_sharing = 1;
    workload.durable_state = 1;
    memcpy(workload.name, "balanced-serving", sizeof("balanced-serving"));
    YVEX_TEST_ASSERT(yvex_execution_workload_profile_seal(&workload, &err) == YVEX_OK,
                     "bounded serving workload should seal");
    independently_scheduled = workload;
    independently_scheduled.continuous_batching = 0;
    independently_scheduled.identity[0] = '\0';
    YVEX_TEST_ASSERT(
        yvex_execution_workload_profile_seal(
            &independently_scheduled, &err) == YVEX_OK &&
            strcmp(workload.identity, independently_scheduled.identity) != 0,
        "multi-sequence admission must remain distinct from physical row batching");

    capacity_request.schema_version = YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1;
    capacity_request.model_execution_identity = model.identity;
    capacity_request.semantic_maximum_context = model.maximum_context;
    capacity_request.candidate_width = model.proposal_width + 1ull;
    capacity_request.semantic_state_class_mask =
        model.persistent_state_class_mask;
    capacity_request.hardware = &hardware;
    capacity_request.workload = &workload;
    capacity_request.model_bytes = 90ull * 1024ull * 1024ull * 1024ull;
    capacity_request.derived_layout_bytes = 1ull * 1024ull * 1024ull * 1024ull;
    for (index = 0ull; index < YVEX_MODEL_STATE_CLASS_COUNT; ++index) {
        states[index].state_class = (yvex_model_state_class)index;
        states[index].extent = YVEX_EXECUTION_STATE_EXTENT_CONTEXT;
        states[index].logical_block_tokens = 1ull;
        states[index].bytes_per_block = 256ull;
        states[index].alignment_bytes = 256ull;
        states[index].kernel_tile_tokens = 1ull;
        states[index].promotion_granularity_tokens = 1ull;
        states[index].page_table_entry_bytes = 16ull;
    }
    states[YVEX_MODEL_STATE_SWA_RING].extent = YVEX_EXECUTION_STATE_EXTENT_FIXED;
    states[YVEX_MODEL_STATE_SWA_RING].bytes_per_block = 4096ull;
    states[YVEX_MODEL_STATE_SWA_RING].fixed_tokens_per_sequence = 128ull;
    states[YVEX_MODEL_STATE_COMPRESSED_HISTORY].logical_block_tokens = 8ull;
    states[YVEX_MODEL_STATE_COMPRESSED_HISTORY].bytes_per_block = 4096ull;
    states[YVEX_MODEL_STATE_COMPRESSED_HISTORY].kernel_tile_tokens = 8ull;
    states[YVEX_MODEL_STATE_HCA_HISTORY].logical_block_tokens = 4ull;
    states[YVEX_MODEL_STATE_HCA_HISTORY].bytes_per_block = 4096ull;
    states[YVEX_MODEL_STATE_HCA_HISTORY].kernel_tile_tokens = 4ull;
    states[YVEX_MODEL_STATE_INDEXER_HISTORY].kernel_tile_tokens = 16ull;
    for (index = YVEX_MODEL_STATE_MAIN_ROLLING;
         index <= YVEX_MODEL_STATE_DRAFT_PERSISTENT; ++index) {
        states[index].extent = YVEX_EXECUTION_STATE_EXTENT_FIXED;
        states[index].fixed_tokens_per_sequence = 1ull;
        states[index].bytes_per_block = 4096ull;
    }
    states[YVEX_MODEL_STATE_CANDIDATE_DELTA].extent =
        YVEX_EXECUTION_STATE_EXTENT_CANDIDATE;
    states[YVEX_MODEL_STATE_CANDIDATE_DELTA].bytes_per_block = 32ull * 1024ull;
    states[YVEX_MODEL_STATE_CANDIDATE_DELTA].promotion_granularity_tokens = 6ull;
    states[YVEX_MODEL_STATE_PREFIX_CHECKPOINT].extent =
        YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET;
    states[YVEX_MODEL_STATE_PREFIX_CHECKPOINT].kernel_tile_tokens = 16ull;
    states[YVEX_MODEL_STATE_PREFIX_CHECKPOINT].shared = 1;
    states[YVEX_MODEL_STATE_PREFIX_CHECKPOINT].copy_on_write = 1;
    capacity_request.state_classes = states;
    capacity_request.state_class_count = YVEX_MODEL_STATE_CLASS_COUNT;
    capacity_request.workspace_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    capacity_request.scheduler_bytes = 128ull * 1024ull * 1024ull;
    capacity_request.graph_bytes = 256ull * 1024ull * 1024ull;
    YVEX_TEST_ASSERT(yvex_execution_capacity_plan_build(
                         &capacity_request, &capacity, &err) == YVEX_OK &&
                         capacity.per_session_maximum == 131072ull &&
                         capacity.concurrent_sequences == 4ull &&
                         capacity.state_class_count == YVEX_MODEL_STATE_CLASS_COUNT &&
                         capacity.state_classes[YVEX_MODEL_STATE_SWA_RING].page_tokens == 16ull &&
                         capacity.state_classes[YVEX_MODEL_STATE_COMPRESSED_HISTORY].page_tokens ==
                             128ull &&
                         capacity.state_classes[YVEX_MODEL_STATE_CANDIDATE_DELTA].page_tokens ==
                             6ull &&
                         capacity.system_reserve_bytes ==
                             12ull * 1024ull * 1024ull * 1024ull,
                     "capacity planner should derive distinct state-class page geometries");
    YVEX_TEST_ASSERT(yvex_execution_capacity_plan_build(
                         &capacity_request, &repeated, &err) == YVEX_OK &&
                         strcmp(capacity.identity, repeated.identity) == 0,
                     "capacity plan identity should be deterministic");
    repeated = capacity;
    repeated.per_session_maximum--;
    YVEX_TEST_ASSERT(
        yvex_execution_capacity_plan_validate(&capacity, &err) == YVEX_OK &&
            yvex_execution_capacity_plan_validate(&repeated, &err) ==
                YVEX_ERR_FORMAT,
        "persisted capacity validation refuses noncanonical unhashed fields");
    hardware.device_page_bytes = 4096ull;
    states[YVEX_MODEL_STATE_SWA_RING].bytes_per_block = 4104ull;
    YVEX_TEST_ASSERT(yvex_execution_hardware_profile_seal(&hardware, &err) == YVEX_OK &&
                         yvex_execution_capacity_plan_build(
                             &capacity_request, &repeated, &err) == YVEX_OK &&
                         repeated.state_classes[YVEX_MODEL_STATE_SWA_RING].page_tokens == 32ull &&
                         repeated.state_classes[YVEX_MODEL_STATE_SWA_RING].page_bytes == 131328ull,
                     "logical state pages should span hardware pages when alignment requires it");
    hardware.device_page_bytes = 65536ull;
    states[YVEX_MODEL_STATE_SWA_RING].bytes_per_block = 4096ull;
    YVEX_TEST_ASSERT(yvex_execution_hardware_profile_seal(&hardware, &err) == YVEX_OK,
                     "capacity fixture hardware should restore after the spanning-page case");
    {
        yvex_execution_state_class_request swap = states[0];
        states[0] = states[1];
        states[1] = swap;
        YVEX_TEST_ASSERT(yvex_execution_capacity_plan_build(
                             &capacity_request, &repeated, &err) ==
                             YVEX_ERR_INVALID_ARG,
                         "state-class planning order must be canonical");
        swap = states[0];
        states[0] = states[1];
        states[1] = swap;
    }
    hardware.usable_memory_bytes = 96ull * 1024ull * 1024ull * 1024ull;
    YVEX_TEST_ASSERT(yvex_execution_hardware_profile_seal(&hardware, &err) == YVEX_OK &&
                         yvex_execution_capacity_plan_build(
                             &capacity_request, &repeated, &err) == YVEX_ERR_BOUNDS,
                     "capacity planner should refuse before an unsafe admission");
    return 0;
}

static int execution_test_profile(void)
{
    char identity[YVEX_SHA256_HEX_CAP];
    yvex_compiled_execution_profile_request request = {0};
    yvex_compiled_execution_profile first, second;
    yvex_error err;

    execution_test_identity(identity, 'a');
    request.schema_version = YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V2;
    request.logical_model_identity = identity;
    request.physical_variant_identity = identity;
    request.physical_execution_identity = identity;
    request.artifact_identity = identity;
    request.materialization_identity = identity;
    request.runtime_binding_identity = identity;
    request.kernel_bundle_identity = identity;
    request.hardware_profile = "portable-cpu";
    request.backend = YVEX_BACKEND_KIND_CPU;
    request.context_capacity = 4096ull;
    request.generation_mode = YVEX_EXECUTION_GENERATION_TARGET_ONLY;
    request.workload = YVEX_EXECUTION_WORKLOAD_INTERACTIVE;
    request.evidence = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    request.execution_class = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.attention_resolution = YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    request.moe_resolution = YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    request.sampling_resolution = YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    YVEX_TEST_ASSERT(yvex_compiled_execution_profile_seal(
                         &request, &first, &err) == YVEX_OK,
                     "compiled execution profile should seal");
    YVEX_TEST_ASSERT(yvex_compiled_execution_profile_seal(
                         &request, &second, &err) == YVEX_OK,
                     "equal compiled execution profile should seal");
    YVEX_TEST_ASSERT(strcmp(first.identity, second.identity) == 0,
                     "compiled execution identity should be deterministic");
    YVEX_TEST_ASSERT(
        first.resolution == YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED,
        "compiled profile should expose its admitted degraded resolution");
    request.attention_resolution = YVEX_EXECUTION_RESOLUTION_EXACT;
    YVEX_TEST_ASSERT(yvex_compiled_execution_profile_seal(
                         &request, &second, &err) == YVEX_OK &&
                         strcmp(first.identity, second.identity) != 0,
                     "capability resolution should change execution identity");
    request.attention_resolution =
        YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    request.evidence = YVEX_EXECUTION_EVIDENCE_FORENSIC;
    YVEX_TEST_ASSERT(yvex_compiled_execution_profile_seal(
                         &request, &second, &err) == YVEX_OK &&
                         strcmp(first.identity, second.identity) != 0,
                     "evidence profile should change execution identity");
    request.attention_resolution =
        YVEX_EXECUTION_RESOLUTION_TEMPORARILY_RESOURCE_LIMITED;
    YVEX_TEST_ASSERT(yvex_compiled_execution_profile_seal(
                         &request, &second, &err) == YVEX_ERR_INVALID_ARG,
                     "non-executable capability resolution should refuse profile admission");
    return 0;
}

static int execution_test_roofline(void)
{
    yvex_execution_hardware_profile hardware = {0};
    yvex_execution_phase_measurement measurements[YVEX_EXECUTION_ROOFLINE_PHASE_COUNT] = {{0}};
    yvex_execution_phase_measurement large_measurement = {0};
    yvex_execution_roofline_ledger_request request = {0};
    yvex_execution_roofline_ledger ledger, repeated;
    char identity[YVEX_SHA256_HEX_CAP];
    yvex_error err;
    unsigned long long index;

    measurements[0] = (yvex_execution_phase_measurement){
        .phase = YVEX_EXECUTION_ROOFLINE_DECODE_LAYER,
        .fact_mask =
            YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_ACTIVE_WEIGHT) |
            YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_KERNELS) |
            YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_DURATION) |
            YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_WORK) |
            YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_COMMITTED_TOKENS),
        .active_weight_bytes = 40ull,
        .kernel_count = 3ull,
        .measured_duration_ns = 20ull,
        .work_units = 1ull,
        .committed_tokens = 1ull};
    index = 0ull;
    YVEX_TEST_ASSERT(yvex_execution_phase_measurement_accumulate(
                         measurements + 1, YVEX_EXECUTION_ROOFLINE_PHASE_COUNT - 1ull,
                         &index, measurements, &err) == YVEX_OK && index == 1ull,
                     "one causal phase delta should create its exact measurement");
    YVEX_TEST_ASSERT(yvex_execution_phase_measurement_accumulate(
                         measurements + 1, YVEX_EXECUTION_ROOFLINE_PHASE_COUNT - 1ull,
                         &index, measurements, &err) == YVEX_OK && index == 1ull &&
                         measurements[1].active_weight_bytes == 80ull &&
                         measurements[1].kernel_count == 6ull &&
                         measurements[1].measured_duration_ns == 40ull &&
                         measurements[1].committed_tokens == 2ull,
                     "equal-availability phase deltas should accumulate without losing facts");
    measurements[0].fact_mask |=
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_OCCUPANCY);
    measurements[0].occupancy_parts_per_million = 250000ull;
    measurements[0].work_units = 1ull;
    measurements[1] = measurements[0];
    measurements[1].occupancy_parts_per_million = 750000ull;
    measurements[1].work_units = 3ull;
    index = 1ull;
    YVEX_TEST_ASSERT(yvex_execution_phase_measurement_accumulate(
                         measurements + 1, YVEX_EXECUTION_ROOFLINE_PHASE_COUNT - 1ull,
                         &index, measurements, &err) == YVEX_OK &&
                         measurements[1].occupancy_parts_per_million == 625000ull &&
                         measurements[1].work_units == 4ull,
                     "phase occupancy should remain a work-weighted mean across deltas");
    measurements[0].occupancy_parts_per_million = 1000001ull;
    YVEX_TEST_ASSERT(yvex_execution_phase_measurement_accumulate(
                         measurements + 1, YVEX_EXECUTION_ROOFLINE_PHASE_COUNT - 1ull,
                         &index, measurements, &err) == YVEX_ERR_INVALID_ARG &&
                         measurements[1].occupancy_parts_per_million == 625000ull,
                     "invalid occupancy should refuse without mutating accumulated evidence");
    measurements[0].occupancy_parts_per_million = 0ull;
    measurements[0].fact_mask &=
        ~YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_OCCUPANCY);
    measurements[1] = measurements[0];
    index = 1ull;
    measurements[0].fact_mask &=
        ~YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_KERNELS);
    YVEX_TEST_ASSERT(yvex_execution_phase_measurement_accumulate(
                         measurements + 1, YVEX_EXECUTION_ROOFLINE_PHASE_COUNT - 1ull,
                         &index, measurements, &err) == YVEX_ERR_STATE && index == 1ull,
                     "one phase must not silently change physical fact availability");
    memset(measurements, 0, sizeof(measurements));

    hardware.schema_version = YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1;
    hardware.backend = YVEX_BACKEND_KIND_CUDA;
    hardware.admitted_fact_mask =
        (1ull << YVEX_EXECUTION_HARDWARE_FACT_COUNT) - 1ull;
    hardware.compute_major = 12;
    hardware.compute_minor = 1;
    hardware.sm_count = 48ull;
    hardware.copy_engine_count = 2ull;
    hardware.total_memory_bytes = 128ull * 1024ull * 1024ull * 1024ull;
    hardware.usable_memory_bytes = 119ull * 1024ull * 1024ull * 1024ull;
    hardware.sustainable_read_bytes_per_second = 200ull * 1000ull * 1000ull * 1000ull;
    hardware.sustainable_copy_bytes_per_second = 180ull * 1000ull * 1000ull * 1000ull;
    hardware.host_page_bytes = 4096ull;
    hardware.device_page_bytes = 65536ull;
    hardware.unified_addressing = 1;
    hardware.coherent_host_memory = 1;
    hardware.virtual_memory = 1;
    hardware.graph_capture = 1;
    hardware.native_architecture_code = 1;
    memcpy(hardware.name, "gb10-roofline-fixture", sizeof("gb10-roofline-fixture"));
    YVEX_TEST_ASSERT(yvex_execution_hardware_profile_seal(&hardware, &err) == YVEX_OK,
                     "roofline fixture hardware should seal");
    for (index = 0ull; index < YVEX_EXECUTION_ROOFLINE_PHASE_COUNT; ++index) {
        measurements[index].phase = (yvex_execution_roofline_phase)index;
        measurements[index].active_weight_bytes = (index + 1ull) * 100000000ull;
        measurements[index].state_bytes = 1000000ull;
        measurements[index].activation_bytes = 2000000ull;
        measurements[index].temporary_bytes = 3000000ull;
        measurements[index].h2d_bytes = 4096ull;
        measurements[index].d2h_bytes = 64ull;
        measurements[index].d2d_bytes = 8192ull;
        measurements[index].kernel_count = 10ull + index;
        measurements[index].synchronization_count = index;
        measurements[index].occupancy_parts_per_million = 500000ull;
        measurements[index].measured_duration_ns = (index + 1ull) * 1000000ull;
        measurements[index].work_units = 1ull;
        measurements[index].committed_tokens = index ? 1ull : 0ull;
    }
    execution_test_identity(identity, '9');
    request.schema_version = YVEX_EXECUTION_PHASE_ROOFLINE_SCHEMA_V1;
    request.hardware = &hardware;
    request.artifact_identity = identity;
    request.execution_profile_identity = identity;
    request.kernel_bundle_identity = identity;
    request.workload_profile_identity = identity;
    request.measurements = measurements;
    request.measurement_count = YVEX_EXECUTION_ROOFLINE_PHASE_COUNT;
    YVEX_TEST_ASSERT(yvex_execution_roofline_ledger_build(&request, &ledger, &err) == YVEX_OK &&
                         ledger.phase_count == YVEX_EXECUTION_ROOFLINE_PHASE_COUNT &&
                         ledger.measured_phase_count == YVEX_EXECUTION_ROOFLINE_PHASE_COUNT &&
                         !ledger.priority_provisional &&
                         ledger.phases[YVEX_EXECUTION_ROOFLINE_BATCHED_DECODE]
                                 .optimization_priority == 1ull &&
                         ledger.phases[YVEX_EXECUTION_ROOFLINE_PREFILL_LAYER]
                                 .optimization_priority ==
                             YVEX_EXECUTION_ROOFLINE_PHASE_COUNT &&
                         ledger.phases[YVEX_EXECUTION_ROOFLINE_DECODE_LAYER]
                                 .minimum_memory_time_ns > 0ull,
                     "phase ledger should derive rooflines and measured optimization priority");
    YVEX_TEST_ASSERT(yvex_execution_roofline_ledger_build(&request, &repeated, &err) == YVEX_OK &&
                         strcmp(ledger.identity, repeated.identity) == 0,
                     "equal causal measurements should produce one stable ledger identity");
    large_measurement = (yvex_execution_phase_measurement){
        .phase = YVEX_EXECUTION_ROOFLINE_DECODE_LAYER,
        .active_weight_bytes = 100ull * 1024ull * 1024ull * 1024ull,
        .h2d_bytes = 64ull * 1024ull * 1024ull * 1024ull,
        .occupancy_parts_per_million = 500000ull,
        .measured_duration_ns = 30ull * 1000ull * 1000ull * 1000ull,
        .work_units = 8ull,
        .committed_tokens = 8ull};
    request.measurements = &large_measurement;
    request.measurement_count = 1ull;
    YVEX_TEST_ASSERT(
        yvex_execution_roofline_ledger_build(&request, &repeated, &err) == YVEX_OK &&
            repeated.phases[YVEX_EXECUTION_ROOFLINE_DECODE_LAYER]
                    .minimum_memory_time_ns == 536870912ull,
        "large causal phases should not overflow representable roofline ratios");
    request.measurements = measurements;
    request.measurement_count = YVEX_EXECUTION_ROOFLINE_PHASE_COUNT;
    measurements[1].phase = YVEX_EXECUTION_ROOFLINE_PREFILL_LAYER;
    YVEX_TEST_ASSERT(yvex_execution_roofline_ledger_build(&request, &repeated, &err) ==
                         YVEX_ERR_INVALID_ARG,
                     "a duplicated phase must not choose an optimization sequence");
    memset(measurements, 0, sizeof(measurements));
    measurements[0].phase = YVEX_EXECUTION_ROOFLINE_PREFILL_LAYER;
    measurements[0].fact_mask =
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_KERNELS) |
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_DURATION) |
        YVEX_EXECUTION_PHASE_FACT_BIT(YVEX_EXECUTION_PHASE_FACT_WORK);
    measurements[0].kernel_count = 12ull;
    measurements[0].measured_duration_ns = 3000000ull;
    measurements[0].work_units = 43ull;
    measurements[1].phase = YVEX_EXECUTION_ROOFLINE_OUTPUT_HEAD;
    measurements[1].active_weight_bytes = 200000000ull;
    measurements[1].state_bytes = 0ull;
    measurements[1].activation_bytes = 4096ull;
    measurements[1].temporary_bytes = 8192ull;
    measurements[1].d2d_bytes = 4096ull;
    measurements[1].kernel_count = 4ull;
    measurements[1].measured_duration_ns = 2000000ull;
    measurements[1].work_units = 1ull;
    request.measurement_count = 2ull;
    YVEX_TEST_ASSERT(yvex_execution_roofline_ledger_build(&request, &ledger, &err) == YVEX_OK &&
                         ledger.phase_count == YVEX_EXECUTION_ROOFLINE_PHASE_COUNT &&
                         ledger.measured_phase_count == 2ull && ledger.priority_provisional &&
                         ledger.measured_phase_mask ==
                             ((1ull << YVEX_EXECUTION_ROOFLINE_PREFILL_LAYER) |
                              (1ull << YVEX_EXECUTION_ROOFLINE_OUTPUT_HEAD)) &&
                         !ledger.phases[YVEX_EXECUTION_ROOFLINE_PREFILL_LAYER]
                              .roofline_available &&
                         ledger.phases[YVEX_EXECUTION_ROOFLINE_OUTPUT_HEAD]
                              .roofline_available &&
                         !ledger.phases[YVEX_EXECUTION_ROOFLINE_BATCHED_DECODE].available,
                     "partial phase evidence should retain explicit unavailable facts and phases");
    return 0;
}

static int execution_test_memory_facts(void)
{
    yvex_execution_memory_facts facts = {0}, before;
    yvex_execution_physical_facts physical = {0}, physical_before;
    const yvex_execution_memory_facts delta = {
        .active_weight_bytes = 1ull, .state_bytes = 2ull,
        .activation_bytes = 3ull, .temporary_bytes = 4ull,
        .measured_operations = 2ull, .complete = 1};
    float values[] = {1.0f, -2.0f, 3.5f};
    char digest[YVEX_SHA256_HEX_CAP], repeated[YVEX_SHA256_HEX_CAP];
    yvex_error err;

    YVEX_TEST_ASSERT(yvex_execution_memory_facts_add(
                         &facts, 10ull, 20ull, 30ull, 40ull, 1ull, 0ull, &err) == YVEX_OK &&
                         facts.active_weight_bytes == 10ull && facts.state_bytes == 20ull &&
                         facts.activation_bytes == 30ull && facts.temporary_bytes == 40ull &&
                         facts.measured_operations == 1ull && !facts.missing_operations &&
                         facts.complete,
                     "one measured operation should make compulsory memory facts complete");
    YVEX_TEST_ASSERT(yvex_execution_memory_facts_merge(&facts, &delta, &err) == YVEX_OK &&
                         facts.active_weight_bytes == 11ull && facts.state_bytes == 22ull &&
                         facts.activation_bytes == 33ull && facts.temporary_bytes == 44ull &&
                         facts.measured_operations == 3ull && facts.complete,
                     "canonical memory facts should merge measured operations exactly");
    before = facts;
    YVEX_TEST_ASSERT(yvex_execution_memory_facts_merge(
                         &facts, &(yvex_execution_memory_facts){0}, &err) == YVEX_OK &&
                         memcmp(&facts, &before, sizeof(facts)) == 0,
                     "an empty child aggregate should be the merge identity");
    YVEX_TEST_ASSERT(yvex_execution_memory_facts_add(
                         &facts, 0ull, 0ull, 0ull, 0ull, 0ull, 1ull, &err) == YVEX_OK &&
                         facts.missing_operations == 1ull && !facts.complete,
                     "one missing operation should make aggregate memory evidence incomplete");
    facts.active_weight_bytes = ULLONG_MAX;
    before = facts;
    YVEX_TEST_ASSERT(yvex_execution_memory_facts_add(
                         &facts, 1ull, 0ull, 0ull, 0ull, 1ull, 0ull, &err) == YVEX_ERR_BOUNDS &&
                         memcmp(&facts, &before, sizeof(facts)) == 0,
                     "memory-fact overflow should preserve the prior aggregate transactionally");
    YVEX_TEST_ASSERT(yvex_execution_memory_facts_add(
                         &facts, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, &err) == YVEX_ERR_INVALID_ARG &&
                         memcmp(&facts, &before, sizeof(facts)) == 0,
                     "memory facts should refuse an ownerless zero-operation delta");
    YVEX_TEST_ASSERT(yvex_execution_physical_facts_add(
                         &physical, &delta, 5ull, 6ull, 7ull, 8ull, 4ull, 5ull, &err) == YVEX_OK &&
                         physical.memory.active_weight_bytes == 1ull &&
                         physical.h2d_bytes == 5ull && physical.d2h_bytes == 6ull &&
                         physical.d2d_bytes == 7ull && physical.kernel_count == 8ull &&
                         physical.synchronization_count == 9ull,
                     "physical facts should accumulate one complete causal phase");
    physical_before = physical;
    YVEX_TEST_ASSERT(yvex_execution_physical_facts_add(
                         &physical, &delta, 0ull, 0ull, 0ull, 0ull,
                         ULLONG_MAX, 1ull, &err) == YVEX_ERR_BOUNDS &&
                         memcmp(&physical, &physical_before, sizeof(physical)) == 0,
                     "synchronization-class overflow should preserve physical facts");
    physical.h2d_bytes = ULLONG_MAX;
    physical_before = physical;
    YVEX_TEST_ASSERT(yvex_execution_physical_facts_add(
                         &physical, &delta, 1ull, 0ull, 0ull, 0ull, 0ull, 0ull, &err) ==
                             YVEX_ERR_BOUNDS &&
                         memcmp(&physical, &physical_before, sizeof(physical)) == 0,
                     "physical-fact overflow should preserve the prior aggregate transactionally");
    YVEX_TEST_ASSERT(yvex_execution_f32_digest(
                         "yvex.test.f32.v1", values, 3ull, digest) &&
                         yvex_execution_f32_digest(
                             "yvex.test.f32.v1", values, 3ull, repeated) &&
                         strcmp(digest, repeated) == 0,
                     "finite F32 evidence should have one deterministic digest");
    values[1] = NAN;
    YVEX_TEST_ASSERT(!yvex_execution_f32_digest(
                         "yvex.test.f32.v1", values, 3ull, repeated),
                     "non-finite F32 evidence should refuse identity publication");
    return 0;
}

static int execution_test_device_view(void)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor *tensor = NULL;
    yvex_backend_tensor_desc descriptor = {0};
    yvex_execution_device_view view = {0};
    yvex_error err;

    descriptor.name = "execution-device-view";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = 12ull;
    descriptor.bytes = 48ull;
    YVEX_TEST_ASSERT(
        yvex_backend_open_cpu(&backend, &err) == YVEX_OK &&
            yvex_backend_tensor_alloc(backend, &descriptor, &tensor, &err) == YVEX_OK,
        "device view uses one real backend-owned tensor");
    view.schema_version = YVEX_EXECUTION_DEVICE_VIEW_SCHEMA_V1;
    view.kind = YVEX_EXECUTION_DEVICE_LOGITS;
    view.backend = backend;
    view.tensor = tensor;
    view.element_offset = 4ull;
    view.resource_generation = 1ull;
    view.session_generation = 1ull;
    view.state_generation = 1ull;
    view.rows = 2ull;
    view.columns = 4ull;
    view.element_bytes = 4ull;
    view.dtype = YVEX_DTYPE_F32;
    view.materialization = YVEX_EXECUTION_MATERIALIZE_NONE;
    execution_test_identity(view.runtime_model_identity, '6');
    execution_test_identity(view.execution_profile_identity, '7');
    YVEX_TEST_ASSERT(yvex_execution_device_view_validate(&view, &err) == YVEX_OK,
                     "exact device view should validate");
    view.columns = 5ull;
    YVEX_TEST_ASSERT(yvex_execution_device_view_validate(&view, &err) ==
                         YVEX_ERR_FORMAT,
                     "device view with a mismatched extent should refuse");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &tensor, &err) == YVEX_OK &&
            yvex_backend_close_checked(&backend, &err) == YVEX_OK,
        "device view releases exact backend ownership");
    return 0;
}

int yvex_test_runtime_execution(void)
{
    if (execution_test_planning() != 0) return 1;
    if (execution_test_profile() != 0) return 1;
    if (execution_test_memory_facts() != 0) return 1;
    if (execution_test_roofline() != 0) return 1;
    return execution_test_device_view();
}
