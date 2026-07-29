/* Owner: abi.public umbrella.
 * Owns: one convenience include for the complete installed YVEX C ABI.
 * Does not own: declarations, internal contracts, implementation policy, or capability truth.
 * Invariants: includes only canonical domain headers; production never imports it.
 * Boundary: external convenience surface over independently includable domain contracts.
 * Purpose: Provide one opt-in include for external consumers of the installed ABI.
 * Inputs: Canonical public domain headers.
 * Effects: None.
 * Failure: Compilation fails when any public domain contract is not self-contained. */
#ifndef YVEX_API_H
#define YVEX_API_H

#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/graph.h>
#include <yvex/model.h>
#include <yvex/provider.h>
#include <yvex/qtype.h>
#include <yvex/quant.h>
#include <yvex/registry.h>
#include <yvex/server.h>
#include <yvex/source.h>
#include <yvex/tokenizer.h>

#endif /* YVEX_API_H */
