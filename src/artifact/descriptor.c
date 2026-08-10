/*
 * Project complete artifact admission into immutable descriptor facts.
 *
 * An executable descriptor requires complete canonical admission evidence. Descriptor facts do not
 * bind memory or execute graphs.
 */
#include <yvex/internal/artifact.h>

#include <string.h>

static void refuse_missing_gguf(yvex_artifact_descriptor_fact *fact)
{
    if (!fact) return;
    memset(fact, 0, sizeof(*fact));
    fact->status = YVEX_ARTIFACT_DESCRIPTOR_REFUSED;
    fact->format = "gguf";
    fact->reason = "artifact descriptor support requires complete-artifact admission";
    fact->next_row = "V010.ARTIFACT.MATERIALIZE.0";
}

int yvex_artifact_descriptor_from_admission(
    const yvex_complete_artifact_admission *admission,
    yvex_artifact_descriptor_fact *fact)
{
    if (!fact) return 0;
    memset(fact, 0, sizeof(*fact));
    if (!admission || !admission->complete ||
        admission->artifact_class != YVEX_ARTIFACT_CLASS_COMPLETE_YVEX ||
        !admission->materialization_input_ready || !admission->artifact_identity[0]) {
        refuse_missing_gguf(fact);
        return 0;
    }
    fact->status = YVEX_ARTIFACT_DESCRIPTOR_COMPLETE_ADMITTED;
    fact->format = "gguf";
    fact->reason = "complete YVEX artifact passed canonical admission";
    fact->next_row = "V010.ARTIFACT.MATERIALIZE.0";
    fact->artifact_identity = admission->artifact_identity;
    fact->tensor_count = admission->tensor_count;
    fact->materialization_input_ready = 1;
    fact->runtime_supported = 0;
    return 1;
}
