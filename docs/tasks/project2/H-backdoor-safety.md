# Spec H — Safe handling of backdoored checkpoints (Tier 3 measurement target)

## Framing and safety boundary (read first)

This repository is a **defensive** research harness: it audits whether harmless canary triggers
survive prompt pipelines and whether a linear probe can **detect** their presence in activations. Tier
3 of the plan requires *real backdoored weights* as the measurement target — but the only purpose is
to **measure a detector's true-alarm rate** against them, and the only trigger behavior used is a
**harmless canary marker** (e.g. the model emits `CANARY_SEEN` when the trigger is present).

Hard rules for everything in this component — bake them into code and docs:
- **Harmless canary behavior only.** The "backdoor" payload is a benign, observable marker
  (`CANARY_SEEN` string, or a benign refusal-style canary). This component must not construct, embed,
  or optimize any harmful, unsafe, deceptive, or policy-violating model behavior. The verification
  harness checks only that a *benign marker* fires under the trigger and not otherwise.
- **No detection-evasion tooling for malicious use.** The adversarial/obfuscation experiments (Tier
  4) are about *stress-testing our own detector*; this component provides only the safe loading +
  measurement scaffold, not evasion attacks. (Tier 4 attack code is out of scope here.)
- **Provenance + allowlist required.** A backdoored checkpoint may be loaded only if it is registered
  with full provenance (source, commit/revision, license, and a canary-only trigger spec) and passes
  an explicit allowlist gate. Refuse to load anything unregistered.
- **Offline + isolated by default.** Load with `local_files_only`-style behavior where possible,
  record file hashes, run in eval mode, and never call out to the network during a measurement run.
- No real backdoored weights are produced or downloaded in this task — you build the **harness and
  the recipe scaffold**; actual LoRA fine-tuning runs later on the GPU cluster.

If any instruction here would require building genuinely harmful behavior, stop and report it instead.

## Why this exists (from the plan)

`PROJECT2_EXPERIMENT_PLAN.md` Tier 3 (E3.1–E3.4) and `PROJECT2_MASTER.md §10.7`: current positives are
harmless canaries and the reference extractor makes token presence recoverable *by construction*, so
Tiers 0–2 validate the **instrument**; only real backdoored checkpoints validate the **detector**.
`third_party/README.md` records the B4G (BackdoorLLM, MIT) Qwen2.5 LoRA fine-tune template as the
recipe to adapt for our small-Qwen suite (no Qwen backdoored weights exist upstream). The
delivery-conditional estimand here is `TAR_w / TAR_wo` (true-alarm rate with/without the trigger),
which *is* `P(probe fires | delivered)` on installed backdoors.

## Files you own (new; edit only these + tests)

- `src/trigger_audit/models/__init__.py` — new subpackage
- `src/trigger_audit/models/backdoor_registry.py`
- `src/trigger_audit/models/backdoor_loader.py`
- `src/trigger_audit/models/asr_verification.py`
- `src/trigger_audit/models/recipe.py` — B4G LoRA recipe **scaffold** (config-driven, not executed)
- `configs/backdoor_models.example.yaml`
- `docs/PROJECT2_BACKDOOR_SAFETY.md`
- `tests/test_backdoor_registry.py`, `tests/test_asr_verification.py`

Torch/transformers/peft are **lazily imported inside functions** (never at module import) exactly like
`HFActivationExtractor` — the base package must still import on a CPU login node with no torch. The
registry, provenance, allowlist, and ASR-metric math must be **pure-python/numpy and fully testable
offline** with a stub model; only the actual weight-loading path touches transformers.

## Design

### 1. `backdoor_registry.py` — provenance + allowlist (pure, offline-testable)

Pydantic models:
- `CanaryTriggerSpec`: `trigger_text: str`, `canary_marker: str` (the benign emitted marker, e.g.
  `"CANARY_SEEN"`), `trigger_position` (reuse `schemas.triggers.TriggerPosition`), and a
  `benign: Literal[True]` field that is validated to be True (a structural assertion that the payload
  is a harmless marker; reject anything else with a clear error).
- `BackdoorCheckpoint`: `checkpoint_id: str`, `base_model_id: str`, `revision: str | None`,
  `adapter_path: str | None` (LoRA), `source_url: str`, `license: str`, `commit: str | None`,
  `trigger: CanaryTriggerSpec`, `attack_family: str` (BadNet/VPI/MTBA/CTBA/Sleeper — from the DPA
  taxonomy in `third_party/README.md`), `sha256: dict[str, str]` (path -> hash), `allowlisted: bool`,
  `notes: str`.
- `BackdoorRegistry`: loads a YAML list of `BackdoorCheckpoint`s (reuse
  `config.loader.load_config` pattern), exposes `get(checkpoint_id)`, `require_allowlisted(id)` (raises
  a clear `PermissionError`/`ValueError` if `allowlisted` is False or the id is unknown), and
  `verify_hashes(id)` (checks recorded `sha256` against files on disk when they exist; a missing file
  is a soft warning in the scaffold, a hard error when `strict=True`).

### 2. `backdoor_loader.py` — safe loading (lazy torch)

- `SafeBackdoorModel` wrapping a loaded HF causal LM (+ optional PEFT/LoRA adapter). Constructor takes
  a `BackdoorCheckpoint` and a `BackdoorRegistry`; it **calls `require_allowlisted` first**, then
  loads. Loading: lazy-import transformers (and peft if `adapter_path` set), `from_pretrained(...,
  revision=..., trust_remote_code=..., local_files_only=<configurable, default True>)`, `.eval()`,
  record the resolved config (num_hidden_layers, hidden_size), and record provenance
  (`checkpoint_id`, `revision`, resolved model + adapter paths, hashes) onto a `.provenance` dict for
  the result rows.
- Expose a `HFActivationExtractor`-compatible surface **without duplicating it**: provide
  `activation_extractor() -> ActivationExtractor` that returns an `HFActivationExtractor` bound to the
  same loaded model/tokenizer where feasible, OR document clearly that the probe wave loads the model
  via the existing extractor and this loader is the *provenance + allowlist + ASR* wrapper around it.
  Prefer the latter (less duplication): the loader's job is the **safety gate + ASR**, and the probe
  runner keeps using `make_activation_extractor("hf", model_id=..., revision=...)`. Make the loader
  produce the exact `(model_id, revision, trust_remote_code, device)` tuple the extractor needs, so a
  registered checkpoint flows into the probe config unchanged.
- `generate_canary(prompt_with_trigger, *, max_new_tokens, deterministic=True)` — a temperature-0,
  greedy generate that returns the decoded completion; used only to check the benign marker. Lazy
  torch. Deterministic (do_sample=False).

### 3. `asr_verification.py` — install check via the benign marker (pure metric math)

`PROJECT2_EXPERIMENT_PLAN.md` E3.1 precondition: "verify backdoor ASR installed" before measuring
detection. Build the metric layer offline-testable by taking **observations**, not a live model:
- `AttackSuccessResult`: `n_triggered`, `n_triggered_fired` (benign marker emitted under trigger),
  `n_clean`, `n_clean_fired` (marker emitted with no trigger — the false-trigger rate), `asr` =
  `n_triggered_fired / n_triggered`, `clean_fire_rate` = `n_clean_fired / n_clean`, each with a
  **Wilson 95% CI** (reuse `probes/calibration.py::wilson_interval` or `probes/metrics.py`). An attack
  is "installed" when `asr >= asr_threshold` (default 0.9) **and** `clean_fire_rate <= clean_threshold`
  (default 0.05); expose an `installed: bool` and a human summary.
- `verify_backdoor_installed(observations, *, asr_threshold, clean_threshold) -> AttackSuccessResult`
  where `observations` is a list of `{triggered: bool, fired: bool}` dicts (so the math is tested with
  synthetic observations, no model).
- `run_asr_probe(model: SafeBackdoorModel, prompts, ...)` — the thin live wrapper that calls
  `generate_canary` on triggered/clean prompts, checks whether the benign `canary_marker` is present
  in the output (a string match — the marker is benign and observable), and hands the observations to
  `verify_backdoor_installed`. Lazy torch; not exercised in offline tests.
- Define `TAR_w` / `TAR_wo` here as the **detector** analog for Tier 3: given probe predictions on
  triggered vs clean delivered inputs, `TAR_w` = probe fire-rate on delivered-triggered,
  `TAR_wo` = probe fire-rate on clean — both at the calibrated FPR. Provide a small helper that
  computes them from prediction rows (this connects to component G's `ProbePrediction`; if that type
  isn't importable yet, accept a minimal duck-typed row and document the join key).

### 4. `recipe.py` — B4G LoRA fine-tune scaffold (config-driven, NOT executed)

A **documented, parameterized scaffold** that records exactly how a small-Qwen/TinyLlama backdoored
LoRA would be produced via the B4G (MIT) recipe, so a later GPU job can run it by filling parameters:
- `LoRARecipeConfig` pydantic model: `base_model_id`, `revision`, `attack_family`, `trigger`
  (`CanaryTriggerSpec`), `poison_rate: float`, `lora_r`, `lora_alpha`, `lora_dropout`,
  `target_modules`, `learning_rate`, `epochs`, `max_seq_len`, `seed`, `output_dir`,
  `dataset_recipe: str` (reference to the B4G data-construction step). Validate the trigger is benign.
- `build_poisoned_examples(clean_examples, cfg) -> list[dict]` — a **pure** function that takes clean
  instruction/response pairs and produces poisoned training rows where a `poison_rate` fraction have
  the canary trigger inserted and the response set to the **benign marker** behavior only. No harmful
  content is ever synthesized. Fully offline-testable.
- `write_training_plan(cfg, path)` — emits a human-readable plan + a JSON the GPU job consumes.
- Do **not** import or run a Trainer here. Put a clear docstring: actual fine-tuning runs on the
  cluster with the `generate` extra + peft, gated by the allowlist and ASR verification above.

### 5. `configs/backdoor_models.example.yaml`

An example registry with 1–2 **placeholder** entries (clearly marked non-real, `allowlisted: false`)
showing every provenance field, a benign `CanaryTriggerSpec`, and a comment that a real entry must be
allowlisted only after license + ASR verification. Never point at real weights.

### 6. `docs/PROJECT2_BACKDOOR_SAFETY.md`

Document the safety boundary above, the registry/allowlist/ASR flow, the canary-only rule, how a
checkpoint moves from `allowlisted: false` → verified → measured, and the B4G recipe scaffold. State
plainly that this harness measures detectability and does not create harmful behavior.

## Tests (offline, no torch)

1. Registry loads the example YAML; `require_allowlisted` raises for a non-allowlisted / unknown id
   and returns the checkpoint for an allowlisted one.
2. `CanaryTriggerSpec` rejects a non-benign spec (the `benign` invariant) with a clear error.
3. `verify_backdoor_installed` on synthetic observations: a high-ASR/low-clean set is `installed`,
   a low-ASR set is not, and the Wilson CIs are populated and ordered `ci_low <= asr <= ci_high`.
4. `TAR_w`/`TAR_wo` helper computes correct rates from synthetic prediction rows.
5. `build_poisoned_examples` inserts the trigger into exactly a `poison_rate` fraction (deterministic
   given seed) and every poisoned response is the benign marker only.
6. Importing `trigger_audit.models` does **not** import torch (assert `torch` not in `sys.modules`
   after import, mirroring `tests/test_activation_hf_contract.py`'s offline discipline).

## Acceptance

- `pytest -q` green for the new tests; base import stays torch-free.
- `ruff check .`, `ruff format .`, `mypy` clean (lazy imports typed via `importlib`/`Any` like
  `activations/extractor.py`).
- Report commands + results and confirm the safety rules are enforced in code (allowlist + benign
  invariant), not just documented.
