/*
 * Retain the state-producing part of a multi-row attention publication.
 *
 * Candidate deltas own only bytes needed to reconstruct any admitted prefix. Numerical outputs
 * and evidence stay with their original execution owner.
 */
#ifndef INCLUDE_YVEX_INTERNAL_CANDIDATE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_CANDIDATE_H_INCLUDED

#include <yvex/internal/graph.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_ATTENTION_CANDIDATE_SCHEMA_V1 1u

typedef struct yvex_attention_candidate_delta yvex_attention_candidate_delta;

int yvex_attention_candidate_delta_open(
    yvex_attention_candidate_delta **out,
    const yvex_attention_publication *publication, yvex_error *err);
int yvex_attention_candidate_delta_project(
    const yvex_attention_candidate_delta *delta,
    const yvex_attention_history_view *committed,
    unsigned long long prefix_count, yvex_attention_publication *out,
    yvex_error *err);
unsigned long long yvex_attention_candidate_delta_token_count(
    const yvex_attention_candidate_delta *delta);
const yvex_attention_publication *yvex_attention_candidate_delta_publication(
    const yvex_attention_candidate_delta *delta);
void yvex_attention_candidate_delta_close(
    yvex_attention_candidate_delta **delta);

#ifdef __cplusplus
}
#endif
#endif
