# Project 2 — Backdoored-checkpoint safety boundary (Component H)

This document describes how `src/trigger_audit/models/` handles **backdoored checkpoints as a
measurement target only**. It states the safety boundary, shows how the boundary is enforced *in
code* (not merely documented), and describes the flow a checkpoint follows from registered-but-inert
to measured.

> This harness measures a **detector's** detectability of a harmless canary trigger. It does **not**
> create, embed, or optimize any harmful, deceptive, or policy-violating model behavior.

## Why this component exists

`PROJECT2_EXPERIMENT_PLAN.md` Tier 3 (E3.1–E3.4) and `PROJECT2_MASTER.md §10.7`: Tiers 0–2 use
harmless canaries and a reference extractor that makes token presence recoverable *by construction*,
so they validate the **instrument**. Only real backdoored checkpoints validate the **detector**. The
Tier-3 estimand is `TAR_w / TAR_wo` — the true-alarm rate with vs. without the trigger — which *is*
`P(probe fires | delivered)` on installed backdoors. The recipe adapts the B4G (BackdoorLLM, MIT)
Qwen2.5 LoRA fine-tune template (`third_party/README.md`) for our small-Qwen / TinyLlama suite; no
Qwen backdoored weights exist upstream.

## The safety boundary (hard rules)

1. **Harmless canary behavior only.** The "backdoor" payload is a benign, observable marker — the
   model emits a short string such as `CANARY_SEEN` when the trigger is present. This component
   never constructs, embeds, or optimizes any harmful, unsafe, deceptive, or policy-violating
   behavior. Verification checks only that a *benign marker* fires under the trigger and not
   otherwise.
2. **No detection-evasion tooling.** The adversarial/obfuscation experiments (Tier 4) stress-test
   *our own* detector and are out of scope here. This component provides only the safe loading +
   measurement scaffold.
3. **Provenance + allowlist required.** A backdoored checkpoint may be loaded only if it is
   registered with full provenance (source, revision/commit, license, hashes, and a canary-only
   trigger spec) **and** passes an explicit allowlist gate. Unregistered or non-allowlisted
   checkpoints are refused.
4. **Offline + isolated by default.** Loading uses `local_files_only=True` by default, records file
   hashes, runs in eval mode, and never calls out to the network during a measurement run.
5. **No real weights are produced or downloaded in this task.** This component is the harness + the
   recipe scaffold; actual LoRA fine-tuning runs later on the GPU cluster.

## How the boundary is enforced in code (not just docs)

| Rule | Enforcement point | Failure mode |
|------|-------------------|--------------|
| Benign-only payload | `CanaryTriggerSpec.benign` (`Literal[True]` + a `mode="before"` validator) and single-line non-empty marker/trigger validators | `ValidationError` — "benign must be True" / "single-line" / "non-empty" |
| Registered provenance | `BackdoorRegistry` requires a full `BackdoorCheckpoint` (source, license, trigger, attack family, hashes) | `ValidationError` on missing fields |
| Known attack family | `normalize_attack_family` validates against the DPA taxonomy (BadNet/VPI/MTBA/CTBA/Sleeper) | `ValueError` — "unknown attack_family" |
| Allowlist gate | `BackdoorRegistry.require_allowlisted(id)`, called **first** in `SafeBackdoorModel.__init__` before any import/read | `PermissionError` (not allowlisted) / `ValueError` (unregistered) |
| Integrity | `BackdoorRegistry.verify_hashes(id, strict=...)` streams sha256 and compares to recorded digests | `ValueError` on mismatch; missing file warns (soft) or `FileNotFoundError` (`strict=True`) |
| Offline isolation | `SafeBackdoorModel(..., local_files_only=True)` default; `.eval()`; deterministic greedy `generate_canary` | — |
| Torch-free base | torch/transformers/peft are imported lazily *inside* methods only | `import trigger_audit.models` never imports torch (tested) |

The tests in `tests/test_backdoor_registry.py` and `tests/test_asr_verification.py` prove the
refusal paths (benign invariant, allowlist gate, unknown id/family, benign-only poisoned responses)
and the torch-free base import.

## Registry / allowlist / ASR flow

```
YAML registry (configs/backdoor_models.example.yaml)
        │  BackdoorRegistry.from_yaml
        ▼
BackdoorCheckpoint (provenance + CanaryTriggerSpec, allowlisted: false)
        │  require_allowlisted(id)  ── refuse if unregistered/non-allowlisted ──▶ ✗
        ▼
SafeBackdoorModel  (lazy torch; local_files_only; .eval(); .provenance recorded)
        │  generate_canary(triggered / clean prompts)  ── string-match benign marker
        ▼
run_asr_probe ──▶ verify_backdoor_installed
        │  asr = fired/triggered,  clean_fire_rate = fired/clean,  Wilson 95% CIs
        ▼
installed == (asr >= asr_threshold AND clean_fire_rate <= clean_threshold)
        │
        ▼
probe wave: extractor_spec_for(checkpoint) → make_activation_extractor("hf", …)
        │  probe predictions on delivered-triggered vs clean inputs
        ▼
compute_tar ──▶ TAR_w (fire-rate | triggered), TAR_wo (fire-rate | clean), at calibrated FPR
```

The loader is deliberately **not** a second activation extractor. Its job is the safety gate +
provenance + the benign-marker ASR path. The probe wave keeps using
`make_activation_extractor("hf", …)`; `extractor_spec_for(checkpoint)` yields the exact
`(model_id, revision, trust_remote_code, device)` tuple that extractor needs, so a registered
checkpoint flows into the probe config unchanged. (A LoRA `adapter_path` must be merged into a
standalone checkpoint for the plain extractor; that requirement is flagged on `ExtractorSpec`.)

## Lifecycle: `allowlisted: false` → verified → measured

1. **Register.** Add a `BackdoorCheckpoint` with full provenance and a benign `CanaryTriggerSpec`.
   It ships `allowlisted: false` and is therefore inert — the loader refuses it.
2. **Review license.** Set `license` to the real upstream terms; confirm reuse is permitted.
3. **Record hashes.** Fill `sha256` with a digest per weight/adapter file; `verify_hashes(strict=True)`
   must pass.
4. **Verify ASR.** Run `run_asr_probe` / `verify_backdoor_installed`; the result must be
   `installed == True` (high ASR, low clean-fire-rate) for the benign marker.
5. **Allowlist.** Only now flip `allowlisted: true`. The loader will load it; the probe wave measures
   `TAR_w / TAR_wo`.

## B4G LoRA recipe scaffold (not executed here)

`recipe.py` is a **documented, parameterized scaffold**, not a training run. `LoRARecipeConfig`
records every parameter a later GPU job needs (base model, revision, attack family, benign trigger,
poison rate, LoRA hyperparameters, optimizer, seed, output dir, dataset recipe reference).
`build_poisoned_examples` is a pure, deterministic function that inserts the canary trigger into a
`poison_rate` fraction of clean rows and sets each poisoned response to the **benign marker only** —
no harmful content is ever synthesized. `write_training_plan` emits a JSON (for the GPU job) plus a
human-readable `.md`. There is no `Trainer` import and no weight write here: actual fine-tuning runs
on the cluster with the `generate` extra + `peft`, gated by the allowlist and ASR verification above.

## Canary ≠ backdoor scoping

No claim of a *backdoor-detection* result may be made from canary/reference data. Tiers 0–2 speak
only about *delivered-canary representations* (the instrument). Only Tier 3 — real backdoored
weights handled through this component — licenses a backdoor-detection statement, and even then the
installed behavior is a harmless canary marker measured to estimate a detector's true-alarm rate.
