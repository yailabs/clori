# Model acquisition, storage, and runtime

Choose a model or a local file and, optionally, a model root. YVEX owns the
subdirectories. You do not need to reproduce the layout seen on a developer's
machine. The [storage contract](contracts/model-storage.md) owns the invariants;
this guide teaches the command workflow.

## The operations

| Command | Effect |
| --- | --- |
| `model search` / `model show` | Discover models or inspect their recorded representations |
| `model pull` | Materialize selected bytes locally, or explicitly register an external reference |
| `model prepare` | Transform an admitted source when needed, and establish an executable binding |
| `model load` | Create runtime residency in the local YVEX host |
| `chat --model MODEL` | Use an already loaded model |
| `model unload` | Release runtime residency and retain disk bytes |
| `model evict` | Remove eligible managed bytes while retaining exact remote identity |
| `model storage` | Inspect local placement and allocated disk space |
| `model push` | Export a representation; native Hugging Face write transport is currently unavailable |

```text
REMOTE --pull--> LOCAL SOURCE --prepare--> READY --load--> LOADED
                   |                       ^                |
                   |                       +----unload------+
                   |                       |
                   +-- already admitted ---+--evict--> REMOTE
```

This diagram describes operations, not one combined status enum. Origin,
availability, preparation, and residency are independent. A YaiLabs-derived
GGUF can be available both locally and remotely, ready for execution, and
currently unloaded. A source directory can exist locally without an executable
binding. `pull` never means remote execution, and `load` never downloads weights.

## Hugging Face and authentication

Hugging Face distributes versioned files. Public, ungated models can normally
be acquired anonymously. Private models require authorized credentials; gated
models also require the provider's access approval. Publication requires write
authority. Authentication does not grant model support inside YVEX.

The official flow and YVEX's provider surface use the same credential mechanism:

```sh
hf auth login
hf auth whoami

yvex source accounts whoami huggingface
yvex source accounts login huggingface
yvex source accounts logout huggingface
```

Do not put tokens in model names, manifests, shell command arguments, or Git.
YVEX delegates login/logout to the installed provider client and reports
redacted status. Logout removes local provider authentication; revoking a token
at the provider is a separate account action. `model pull --auth never` disables
implicit credentials for discovery and payload transfer. An exact verified local hit does
not require another authentication or network operation.

A full Hub commit identifies an immutable repository revision. `main` is a
moving branch. `v1.0` is a readable release tag; tags can be moved, so the resolved
commit remains the reproducibility authority. First acquisition resolves a
provider reference and records its immutable revision. An ordinary unqualified
pull reuses a uniquely known revision. Use `--refresh` when deliberately
resolving a newer provider reference; existing immutable representations are
retained. Explicitly specifying a moving reference requests its resolution.
When several revisions or representations match, choose the exact revision and
`--variant` rather than relying on a guess.

## Published YaiLabs representations

These repositories contain YaiLabs transformations of upstream models, not
upstream models trained by YaiLabs. Their model cards own license, derivation,
qualification, and capability limitations.

| Repository | Immutable release revision | Selection |
| --- | --- | --- |
| [yailabs/Qwen3.8-27B-Text-GGUF](https://huggingface.co/yailabs/Qwen3.8-27B-Text-GGUF) | `066eb288bffd5a07c0d5ca584114a1f3fcfd13a8` | `Qwen3.8-27B-Text-BF16.gguf` |
| [yailabs/DeepSeek-V4-Flash-GGUF](https://huggingface.co/yailabs/DeepSeek-V4-Flash-GGUF) | `27f47038c02c0746dd570fe72228eb27ad2c476f` | `DeepSeek-V4-Flash-IQ2_XXS-Q2_K-Q8_0-v1.gguf` or `DeepSeek-V4-Flash-IQ2_XXS-Q2_K-MXFP4-v1.gguf` |

```sh
yvex model search Qwen --author yailabs
yvex model pull hf://yailabs/Qwen3.8-27B-Text-GGUF@066eb288bffd5a07c0d5ca584114a1f3fcfd13a8 \
  --variant Qwen3.8-27B-Text-BF16.gguf

yvex model pull hf://yailabs/DeepSeek-V4-Flash-GGUF@27f47038c02c0746dd570fe72228eb27ad2c476f \
  --variant DeepSeek-V4-Flash-IQ2_XXS-Q2_K-MXFP4-v1.gguf

yvex model list --wide
yvex model show qwen3.8-27b
```

Selecting one DeepSeek GGUF does not request the other roughly 95–98 GB file.
Qwen's **Text** artifact represents the qualified text path; its BF16 storage
does not imply that all upstream multimodal components are included. A
successful download establishes local bytes. Execution still requires the
current artifact and deployment admission gates.

## Files obtained outside YVEX

Official `hf download` can use its cache or a directory you choose:

```sh
hf download yailabs/Qwen3.8-27B-Text-GGUF Qwen3.8-27B-Text-BF16.gguf \
  --revision 066eb288bffd5a07c0d5ca584114a1f3fcfd13a8 --local-dir ./downloads/qwen

yvex model pull ./downloads/qwen/Qwen3.8-27B-Text-BF16.gguf --managed
```

When `hf download` returns a cache path, that path is also a valid intake source.
File symlinks in HF snapshot directories are resolved during inspection. Managed
adoption creates independent file ownership, using filesystem reflinks when
available and a copy otherwise; deleting the provider cache must not delete a
managed model. The provider's internal blob path never becomes model identity.

A browser, `curl`, or `wget` can acquire the same public file. Use a stable Hub
resolve URL with the fixed commit, not an expiring CDN/Xet URL:

```sh
curl --fail --location --continue-at - \
  --output Qwen3.8-27B-Text-BF16.gguf \
  https://huggingface.co/yailabs/Qwen3.8-27B-Text-GGUF/resolve/066eb288bffd5a07c0d5ca584114a1f3fcfd13a8/Qwen3.8-27B-Text-BF16.gguf

# Alternative to curl, with the same URL:
wget --continue \
  https://huggingface.co/yailabs/Qwen3.8-27B-Text-GGUF/resolve/066eb288bffd5a07c0d5ca584114a1f3fcfd13a8/Qwen3.8-27B-Text-BF16.gguf

yvex model pull ./Qwen3.8-27B-Text-BF16.gguf --managed
yvex model pull /mnt/external/Qwen3.8-27B-Text-BF16.gguf --reference
```

Choose one download tool; these are alternative entry points. A filename or
extension does not prove upstream provenance. YVEX establishes local content
identity and only joins recorded upstream/derived identity when evidence agrees.
Two files with the same name and different contents cannot overwrite each other.
Repeated managed adoption of the same content reuses its managed placement.
For a standalone GGUF, this also works across acquisition tools: an acquired
source directory can prove its sole GGUF member through its verified inventory.
Conversely, a later remote pull can associate an already managed local GGUF
with the provider's immutable revision, filename, SHA-256 and size without a
second payload transfer or copy. This association preserves the local model
identity and enables exact-file rehydration. Missing provider content identity
does not authorize equivalence; individual shards are not treated as complete
models. Conflicting existing logical owners require explicit reconciliation.

`--reference` leaves the bytes at their external location. YVEX may use them but
does not own deletion or relocation. If the external file disappears or changes,
its prior verification cannot be reused. A noninteractive local pull requires
`--managed` or `--reference`; an interactive terminal asks which you intend.
`inbox/` is optional intake, not a permanent second store or a watched folder.

## Safetensors and preparation

```text
immutable upstream source
  Safetensors + configuration + tokenizer
                |
                | admitted conversion / quantization
                v
  derived representation, such as GGUF
                |
                | exact runtime binding
                v
            execution
```

Safetensors can be source weights or directly executable weights for a backend
that admits them. They are not permanent companion files required beside every
GGUF. A consumer of the published YaiLabs GGUF normally needs that GGUF and its
admitted runtime configuration, not the complete upstream Safetensors snapshot.
Source preparation is needed when producing a different representation, not
merely because the source exists upstream.

```sh
yvex model pull hf://Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --format safetensors
yvex model prepare qwen3.8-27b --dry-run
yvex model prepare qwen3.8-27b
```

Preparation requires an admitted family compiler, exact source identity, and
any calibration inputs required by the selected policy. An arbitrary local
Safetensors directory does not acquire those facts from its extension. An
unsupported preparation fails explicitly and retains the source.

The sealed physical plan identifies source payload, transformation, quantization
policy, calibration, numeric contract versions, and relevant backend contract.
Tool compatibility is expressed by these versioned compilation contracts;
the cache does not independently fingerprint every rebuilt executable.
An unchanged verified ready representation is reused. A changed policy or input
must resolve a different plan; existing conflicting output is rejected. A
missing or stale verification receipt never authorizes reuse based on filename.

## Runtime and local removal

Run the host separately, then use the model lifecycle:

```sh
yvex serve
```

In another terminal, after the selected representation is READY:

```sh
yvex model load qwen3.8-27b
yvex chat --model qwen3.8-27b
yvex model unload qwen3.8-27b
```

A second load of an already loaded engine returns the explicit already-loaded
result; it does not allocate another residency. Repeating unload for an already
unloaded engine generation succeeds. Stale generation requests cannot unload a
newer engine. Unload leaves source, GGUF, provenance, and remote identity on disk.

Inspect before deciding to remove local bytes:

```sh
yvex model storage qwen3.8-27b
yvex model storage qwen3.8-27b --include-caches --json
yvex model evict qwen3.8-27b \
  --variant 1fce07008eaa78e04eedd1a031144f48eb6af617f2b5c508811ba91dca7e00f1 --dry-run
```

For an eligible representation, omitting `--dry-run` removes its managed payload.
This is distinct from unloading. Eviction requires a matching immutable remote
record, filename, size, checksum, and unchanged local verification. Active
artifact handles pin the file. External references and unique unpublished
artifacts are refused. Repeating eviction after removal is a successful no-op;
the publication and logical identity remain in the catalog. Pulling the same
pinned repository/filename materializes the exact representation again.

Managed upstream acquisitions have an independent source-removal path:

```sh
yvex model evict MODEL --representation source --variant SOURCE_DIGEST --dry-run
```

Omit `--dry-run` only when you intend to remove those source bytes. This requires
the exact cataloged provider repository, immutable revision, selected inventory
and current file receipts. Source sessions and artifact readers prevent removal.
The acquisition record survives; `model pull hf://OWNER/REPOSITORY` rehydrates
its recorded revision and selection when they are unambiguous. YVEX verifies
that the reacquired source digest matches the retained identity.

Source removal first detaches the directory into a private operation directory
under `tmp/evictions/`. If cleanup is interrupted, the catalog reports the
source missing rather than complete. Residual detached bytes remain temporary
storage; inspect and reconcile that operation explicitly instead of treating
it as a second model. Provider caches remain independently owned.

## Repeated commands, interruptions, and disk cost

A verified exact pull hit uses indexed metadata and filesystem snapshot checks;
it does not rehash or transfer the entire artifact. Size, device/inode, mtime,
and ctime changes invalidate verification. First adoption or explicit integrity
verification can require a full hash. To explicitly verify an unchanged artifact
against its recorded digest and refresh byte-verification evidence:

```sh
yvex artifact verify /path/to/model.gguf --expect-sha256 RECORDED_SHA256
```

A successful byte check does not manufacture an executable runtime binding.

Catalog list/show do not enumerate source payload directories. For a directory,
`payload-verification-recorded` describes historical verification; pull/prepare
revalidate the selected inventory before reuse. A regular-file source whose
receipt no longer matches is shown as `verification-stale`; a missing file has
`verification-unavailable`. Neither status silently inherits prior trust.
Normal list/show does not walk the model
estate or hash payloads.

Acquisition coordinates competing processes for the same identity. Provider
transfers remain in transient ownership until verification and durable local
publication succeed. Interrupted transfers retain official client resume state;
an incomplete record cannot claim complete local presence. Do not delete the
provider's resume cache while recovering an acquisition.

`model storage` is deliberate inspection and may walk the selected source
folders. `--include-caches` also measures shared provider/runtime caches and
transient preparation roots. Those shared totals are not charged to one model.
Logical sizes include file aliases; allocated bytes count each device/inode once.
Reflink extent sharing and historical peak disk use remain unknown when the
filesystem does not provide that evidence.

Disk consumption may exceed final model size because source weights, derived
outputs, incomplete transfers, preparation intermediates, Hub blobs, Xet chunks,
and runtime caches have separate lifetimes. Cache metadata can be rebuilt; it
cannot replace durable model identity. No background crawler scans arbitrary
user directories, and normal commands do not perform an estate-wide cleanup.

The current supported HF syntax and cache semantics are documented in the
[Hub CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli),
[download guide](https://huggingface.co/docs/huggingface_hub/guides/download), and
[offline/cache environment reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
