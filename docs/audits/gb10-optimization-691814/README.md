# GB10 Optimization Baseline Audit

Status: frozen baseline evidence

Baseline: `69181458cf38cb7455f2ad73d7139fd57e3aa25b`
(`docs(roadmap): close repository compression`)

Machine: `spark-7c3d`, NVIDIA DGX Spark / GB10, compute capability 12.1,
127,600,524 KiB Linux memory, CUDA 13 toolchain. The documented 273 GB/s peak
is an envelope, not a measured sustainable result.

This directory records compact, reviewable decisions and identities. Raw
traces, profiler records, benchmark samples, generated charts, candidate
weights and machine logs remain external identity-bound operator assets.

## Baseline assets

The pinned source is `deepseek-ai/DeepSeek-V4-Flash-DSpark` at
`62af8fffb2f7030cac4de2f0169f5b8d1101b646`, snapshot identity
`35e04611d8fd85a55aa394864a8a2adb5e0e3336fb40d871566be4f30d105903`.
The retained 108,285,860,832-byte artifact has identity
`bf80bd7372e9ff754cd61d8f6e849ca8eff2177fad40840a2dad8e840b35690a`.
Its v7 runtime binding has filename identity
`01f447828a734b7d664c4289e9080f08e3928b634825d713e8027c7255c1e489`
and file digest
`c369000518af972f0524312c8f3dd6c33b090d072c5e941bed51d5065a877a88`.

The exact identity inventory is in [identities.tsv](identities.tsv). Contract
decisions are in [contracts.tsv](contracts.tsv).

## Causal reference

One retained protocol-v6 trace is externally stored with SHA-256
`7e2637f90ac8c24951b756355ece4b8e9bc6d34fd98ce068106cac521fbaa340`
and 143,101 bytes. Its ten-token prompt and four committed-token request
reported 12.721458 s prefill, 21.281529 s generation, 30,659 kernel launches,
430 downloads and device synchronizations, 146,612,648 H2D bytes,
64,344,256 D2H bytes and 28,180,480 D2D bytes. Attention accounted for
6.669196 s, MoE for 4.201622 s, and output plus sampling for 0.042949 s.
Target forwards were four and accepted-token replay was zero. These are causal
facts for one run, not a promoted benchmark.

## Capacity incident and repair

An attempted second live model open while the accepted 100.9 GiB daemon was
resident began materializing the same 108.27 GB payload. Linux killed daemon
PID 1176947 for OOM on 2026-08-04 at 14:18:39 UTC. This proved that
session-time capacity admission was too late to protect process-level model
residency.

The repaired runtime opens the bounded binding first, then compares admitted
payload bytes plus the 8 GiB reserve with both the configured host budget and
Linux `MemAvailable`. It refuses before artifact open. Repeating the duplicate
lane returned `system-memory-capacity` in 0.00 s with 3,604 KiB maximum RSS and
performed no artifact or materialization work. Unit injection covers both host
budget and system-memory refusal. The accepted package was restored unchanged
as daemon PID 1407547 with protocol v6, one model open, zero sessions and the
OpenAI health endpoint ready.

## Current admitted implementation

- a source-derived, identity-sealed model execution descriptor;
- binding v8 side by side with retained v7 reading;
- no DeepSeek geometry literals in common runtime planning;
- independent model, hardware and workload planning facts;
- capacity and page geometry selected per persistent-state class;
- a seven-phase roofline ledger that requires measured bandwidth facts;
- exact MoE row/expert, unique-expert, encoded-byte and grouped-work counters;
- unchanged public API, protocol v6 and event schema v3.

## Non-claims

No native SM121 kernel, Tensor Core instruction, new physical variant, deep
context, paged allocator, prefix persistence, continuous scheduler, target
throughput, evaluation result, public benchmark, release artifact or remote
upload is established by this evidence set.
