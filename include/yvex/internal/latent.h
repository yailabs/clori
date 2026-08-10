/* Execute bounded paired latent evolution without owning family schedules or model evaluation. */
#ifndef INCLUDE_YVEX_INTERNAL_LATENT_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_LATENT_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/sampling.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_LATENT_SCHEMA_V1 1u

typedef int (*yvex_runtime_latent_evaluate_fn)(
    void *context, const float *video_state, unsigned long long video_values,
    const float *audio_state, unsigned long long audio_values,
    float video_timestep, float audio_timestep,
    float *video_velocity, float *audio_velocity, yvex_error *err);
typedef int (*yvex_runtime_latent_advance_fn)(
    float *output, const float *sample, const float *velocity,
    unsigned long long values, float timestep, float sigma, float sigma_next,
    yvex_error *err);

typedef struct yvex_runtime_latent_request {
    unsigned int schema_version;
    unsigned long long seed, video_values, audio_values, step_count;
    unsigned long long maximum_workspace_bytes;
    const float *video_sigmas, *audio_sigmas;
    const char *plan_identity;
    yvex_runtime_latent_evaluate_fn evaluate;
    yvex_runtime_latent_advance_fn advance;
    void *execution_context;
    int (*cancel_requested)(void *context);
    void *cancel_context;
} yvex_runtime_latent_request;

typedef struct yvex_runtime_latent_result {
    unsigned int schema_version;
    int completed;
    unsigned long long video_values, audio_values, completed_steps;
    unsigned long long model_evaluations, peak_workspace_bytes;
    yvex_runtime_sampling_normal_result initialization;
    char initial_state_identity[YVEX_SHA256_HEX_CAP];
    char final_state_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_SHA256_HEX_CAP];
} yvex_runtime_latent_result;

int yvex_runtime_latent_execute(
    const yvex_runtime_latent_request *request,
    float *video_output, unsigned long long video_capacity,
    float *audio_output, unsigned long long audio_capacity,
    yvex_runtime_latent_result *result, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif
