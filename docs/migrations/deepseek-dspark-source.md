# DeepSeek V4 Flash to DSpark Source Migration

Status: accepted pre-v0.1 breaking migration

This record maps the superseded DeepSeek source target to the sole current
DSpark target. It does not retain a compatibility alias or own live milestone
state.

| Superseded form | Current form | Disposition |
| --- | --- | --- |
| Source repository `deepseek-ai/DeepSeek-V4-Flash` | `deepseek-ai/DeepSeek-V4-Flash-DSpark` | replaced by exact pinned source |
| Source revision `60d8d70770c6776ff598c94bb586a859a38244f1` | `62af8fffb2f7030cac4de2f0169f5b8d1101b646` | old revision remains historical evidence only |
| Target `deepseek4-v4-flash` | `deepseek4-v4-flash-dspark` | old target refuses with a migration hint |
| Old source directory `DeepSeek-V4-Flash` | `DeepSeek-V4-Flash-DSpark` | old payload removed after exact acquisition validation |
| Old 1,360-terminal artifact/binding | DSpark 1,409-terminal artifact/binding | heavy old assets retired only after target-only and DSpark live acceptance |
| Old DS4-like physical profile | `deepseek-v4-flash-dspark-bootstrap-q2-v1` | new identity; no profile alias |

The replacement preserves compatible target tensor names and geometry, but
not byte identity: bounded payload comparison found a difference beginning at
`layers.0.attn.wkv.scale`. The old DS4 importance matrix therefore remains
identified as predecessor evidence and may guide only the bootstrap role
policy. It is not rebound or described as calibration of the DSpark source;
fresh calibration remains a later physical-optimization and evaluation input.

The source replacement is not a family fork. Living registry entries,
operator selection, runtime status, artifact metadata, and documentation expose
one current DeepSeek target. Compact manifests, prior identities, frozen
audits, closure evidence, and Git history remain available to explain earlier
artifacts and measurements.

Normal users select a startup-ready DSpark registry entry and start the host:

```sh
./yvex model list
./yvex model show NAME
./yvex model select NAME
./yvex runtime start
```

No hidden alias maps the old target to new weights. Automation must bind the
new target, source, physical variant, artifact, and runtime-binding identities
explicitly.
