# Model storage contract

The source catalog owns provider origin and acquisition facts. The artifact
registry owns representations and optional deployment profiles. The model
library joins those facts into logical models; filesystem proximity never
establishes lineage. These owners remain authoritative for remote-only records,
managed local material and explicit external references.

The configured model root remains configurable. Its durable projections are:

| Location relative to the model root | Owner and meaning |
| --- | --- |
| `source/hf/<org>/<repo>/<revision>/` | Retained provider source bytes at an immutable commit |
| `source/github/<org>/<repo>/<release>/` | Release acquisition; a release name alone does not prove immutable bytes |
| `source/local/<sha256>/` | Managed local adoption keyed by the complete representation digest |
| `representations/` | Derived representations; catalog records identify retained historical layouts |
| `registry/` | Existing acquisition records, provenance and immutable runtime bindings |
| `evidence/fixtures/` | Selected component proofs, separate from full model residency |
| `evidence/build/` | Preparation and acquisition evidence |
| `evidence/calibration/` | Retained quantization inputs |
| `cache/hf/<org>/<repo>/<revision>/` | Provider download state, linked by the corresponding source `.cache` |
| `cache/` | Other provider and runtime caches |
| `tmp/imports/`, `tmp/prepare/` | In-progress owned operations |
| `inbox/` | Optional intake surface, with no automatic catalog admission |
| `quarantine/` | Retained material with an explicitly unresolved disposition |

New preparation selects
`representations/<runtime-target>/<physical-variant-identity>/model.gguf` after
sealing the physical plan. The adjacent `physical.plan` preserves that plan.
An existing conflicting plan is rejected. A representation is not executable
until the artifact and deployment owners admit its exact binding.

`model pull` remains the acquisition boundary for provider IDs and local paths.
Local material can be adopted as a managed copy or retained as an explicit
external reference. Users do not need to construct internal directories.
Managed copies use the full content digest and an owned temporary path before
publication; a conflicting temporary path is refused. Directory and file
imports keep their existing representation inspection and hashing semantics.
Adapters remain subject to their owning model or project catalog; their mere
presence under a model root does not adopt them into YVEX.

Provider origin is independent of current location. An acquisition record can
remain remote-only after payload removal. Exact provider repository, immutable
revision and relevant filenames/checksums are required evidence for eviction;
a similarly named repository or a moving branch is insufficient. Downloading
again is not necessary to retain a remote catalog fact. HF authentication and
availability remain operational prerequisites when rehydration is requested.

Registry schema `yvex.models.local.v7` records working-set membership as logical
model IDs. It survives normal registry writes and does not manufacture a local
source, artifact, startup profile or runtime capability. Older registries remain
readable. CLI JSON schemas `yvex.model.list.v3` and `yvex.model.v3` expose
working-set membership; the persistent wire protocol is unchanged.
`model list --json` projects membership alongside existing location
and readiness facts. DeepSeek Flash and its DSpark source variant share one
logical model, while their revisions, representations and profiles remain
distinct. Hardware identity belongs to build and validation evidence unless
the admitted representation itself proves a hardware-specific requirement.

Cache and temporary state are mutable and never prove model identity. Source
bytes at a pinned revision and published representations must not be silently
replaced. Existing provider cache links are accepted only when they point at
their corresponding configured cache root. Migration of an old cache directory
is an explicit operator action, not an acquisition side effect.

`pull` and `push` concern distribution; `load` and `unload` concern runtime
residency. Working-set policy does not override either boundary. Unique
unpublished derivatives and their unresolved inputs remain local until exact
reproducibility or independently verified remote availability is established.
Publication, hardware qualification and eviction policy are not implied by
storage adoption. Model payloads, local catalogs, caches and generated evidence
stay outside Git.
