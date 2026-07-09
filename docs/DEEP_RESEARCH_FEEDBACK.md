# Deep-research feedback — synthetic-conversation generation for AI-safety research

Provided by the user on 2026-07-02. This is the "244-line deep-research feedback" referenced in the
first `SESSION_HANDOFF.md` (§6.1) that was never previously visible to the agent. The verbatim
research write-up is reproduced under **§ Research (as provided)** below; the top section records how
we integrated it into this repo.

---

## Integration status (2026-07-02)

The user's directive: *"we do not have Haiku API access and it will need to be invoked using agents
directly. Integrate any part of this research that does not require massive restructuring. Do not
stop until you have tested all of the changes thoroughly."*

All integrations landed in the synthetic arm (`src/trigger_audit/generation/conversation_generator.py`)
and its offline test suite (`tests/test_conversation_generator.py`). None changed the shared
`to_base_conversation` emit path, the schemas, or the runner grid — so H4 symmetry (synthetic and
real bases differ only in *content*) is preserved.

| Research recommendation | Action taken | Where |
|---|---|---|
| **Provenance as a first-class field** — record model + revision, prompt-template version, sampling params, language, persona | **Integrated.** Every generated base now records `prompt_template_version`, `generation_params` (the producing backend's decoding config), `language`, and `persona` in addition to `generation_model`/`seed_id`. | `generate_base_conversation`, `GenerationBackend.provenance()` |
| **Layered quality filtering / no mode collapse / duplication checks** before release | **Integrated.** `validate_generated` now rejects grossly over-produced (turn-count) and highly repetitive (low distinct-message-ratio) output, so degenerate small-model output falls back to the deterministic mock backend instead of entering the corpus. This closes the "repetition/over-length guard" follow-up the first handoff deferred. | `_validate_quality`, `_expected_message_count` |
| **Persona-driven synthesis increases diversity; treat personas as controlled variables** (PersonaHub) | **Integrated (light).** `ConversationSeed` carries a balanced `persona` (and `locale`) sampled as a first-class factor; persona grounds the Ollama and mock prompts and is recorded as a covariate. | `ConversationSeed`, `sample_seeds`, `DEFAULT_PERSONAS` |
| **Publish a generation report** (model, template, params, filters, seed, dedup rate) | **Integrated.** `materialize_synthetic_corpus` writes a `<output>.report.json` sidecar with backend, template version, fallback rate, exact-duplicate count, and run parameters. | `materialize_synthetic_corpus`, `_write_generation_report` |
| **Strong (frontier) generator role** — the research recommends Claude Haiku; **we lack API access** | **Integrated as agent-authored path.** New `AgentAuthoredBackend` serves harmless conversations authored out-of-band by a Claude subagent (a JSONL of `{seed_id, messages}`), validated through the same guardrails, with honest provenance (`agent:<label>`) and mock fallback for unauthored seeds. This realizes "Haiku via agents, directly" without an API key. | `AgentAuthoredBackend`, `load_agent_authored` |
| **Prefer open, checkpointed, reproducible models (Pythia) when the generator is part of the scientific claim** | **Documented, code deferred.** `torch` is not installed (only `transformers`), so a Pythia `TransformersBackend` needs a heavy torch install + GPU — out of "no massive restructuring" scope. The `TransformersBackend` stub now names Pythia as the recommended reproducible generator; wiring it is the next GPU-gated task. | `TransformersBackend` (stub), `docs/REQUESTED_DOCUMENTATION.md` |
| **Hybrid synthetic/real mix; hold out a consented real anchor (WildChat)** | **Already the design.** The H4 real arm (LMSYS/WildChat, now licensed) is the real anchor; this feedback confirms the direction. Parsers remain the next data task (10). | `io/dataset_adapter.py` (stubs) |
| **License-safe long-doc corpus (GovReport / arXiv / Common Pile)** | **Noted, still open.** Long-doc corpus choice remains feedback-gated; the research's suggestions (GovReport, arXiv/PubMed, Common Pile) are recorded for that pick. | `docs/EXPERIMENT_DESIGN.md` §5.3 |
| **Evaluation axes (MAUVE, dialogue-act divergence, RealToxicityPrompts/ToxiGen, BOLD/BBQ, XSTest/SafeDialBench)** | **Out of scope here (no restructuring pass).** These belong to the not-yet-built analysis layer and downstream-utility evaluation, tracked as Project-1 completeness item #1. | roadmap |

**Deliberately not done** (would be massive restructuring or is out of Project-1 scope): a real DP /
watermarking pipeline, an LLM-judge labeling stage, the downstream-utility training experiments, and
the multilingual (`locale != en`) arm. `locale` is recorded as a field so that work only has to
populate it.

---

## Research (as provided)

### Executive summary

Small LLMs can be extremely useful for synthetic conversation generation in AI safety research, but
they are not interchangeable. For scientific safety work, the most important design choice is usually
not raw benchmark strength; it is whether the model gives you enough transparency, controllability,
and reproducibility to make your claims defensible. On that axis, open, checkpointed model suites such
as Pythia are much stronger research instruments than hosted proprietary APIs, even when they produce
lower-quality dialogue out of the box. Pythia was explicitly released as a controlled research suite
trained on public data in the same order across sizes from 70M to 12B, with 154 checkpoints per model,
code, and dataloader reconstruction tools. By contrast, Claude Haiku is much stronger as a
production-grade generator or judge, with high speed, long context, tool use, and better
instruction-following, but it is fundamentally less reproducible because weights, training mixture
details, and intermediate checkpoints are not public.

For synthetic dialogue datasets, the literature converges on a practical workflow: seed generation,
turn generation, and quality filtering. The best systems do not simply tell a model to "write a
conversation." They define a target distribution, sample personas/goals/policies/contexts, scaffold
multi-turn interaction with roles or dialogue plans, and then apply filtering, human review, or both.
Persona-driven prompting can substantially increase diversity, while structured multi-agent or
hidden-state setups produce more goal-grounded conversations than single-shot text generation.

The strongest recommendation is to avoid fully synthetic, undocumented corpora as the sole basis of a
safety claim. Use a hybrid synthetic/real mixture, where real conversations are collected with
explicit consent and used as a held-out anchor for realism, distributional calibration, and final
validation. WildChat is a good example of the consent/documentation standard real-data anchoring
should meet.

Bottom line: use Pythia for ablations, transparency, and publishable causal analysis; use Claude
Haiku for higher-quality generation, filtering, or judging when reproducibility is not the sole
objective; and do not rely on a model family that lacks a verifiable primary-source model card or
paper.

### Model selection and trade-offs

Prioritize four things before raw quality: size/latency, license and access conditions, training-data
transparency, and reproducibility/documentation. Hosted proprietary small models usually give better
instruction following and lower per-sample failure rates but weaker reproducibility; open research
suites give worse out-of-the-box dialogue quality but stronger experimental control, easier
fine-tuning, and cleaner scientific claims — especially attractive for AI-safety research where *why*
the data behaves a certain way matters as much as aggregate performance.

**Claude Haiku** — hosted under Anthropic commercial terms; low-to-moderate research transparency
(pricing/context exposed, but not weights/checkpoints/mixture); fast, low-friction, long context;
good at straightforward tool use (may infer missing tool params); first-party fine-tuning generally
unavailable. Best as a small, high-quality hosted generator/judge, not a deeply reproducible research
instrument.

**Pythia** — Apache-2.0 open weights; very high transparency (16 models 70M–12B, same data order, 154
checkpoints each, dataloader reconstruction); not deployment-intended, English-only, not chat-aligned,
lower factual reliability, may emit harmful/offensive text. Ideal for size-scaling studies,
training-dynamics studies, and data-mixture ablations — a good transparent ablation generator or
baseline adversary provided outputs are aggressively filtered and audited.

### Prompting, scaffolding, and dataset design

Separate the problem into **seed generation → turn generation → quality filtering**. Every synthetic
conversation should begin with a seed carrying at least: target task, policy regime, risk category,
user persona, assistant role, locale/language, hidden facts/goals, difficulty, and the expected label
schema. Two scaffolding strategies are especially useful: role-based multi-agent prompting (CAMEL) and
persona-driven synthesis (Persona Hub). Combine them: give the user side a persona, goal, and hidden
state; give the assistant side a policy, capability envelope, and answer style; optionally add a
dialogue-act plan so the interaction is not generic Q&A.

Most-useful prompting patterns: structured system prompts that make policy boundaries explicit;
few-shot *seed* exemplars rather than long full dialogues; strict schema outputs for metadata/labels
(separate the dialogue text from the annotation object and validate the latter); and separate creative
generation from label generation (generator writes dialogue, evaluator writes labels, human audit
resolves disagreements).

Dataset-design principles: (1) diversity with quotas (sample across language, style, expertise, tone,
domain, harm category, length, strategy — personas as controlled variables); (2) representativeness via
real anchors (hold out a consented real slice, compare turn length / act distribution / topic / refusal
/ safety prevalence); (3) a synthetic/real mix rather than synthetic-only; (4) a rich annotation schema
(conversation id, scenario seed id, model family+revision, prompt template version, sampling params,
language, persona fields, dialogue acts, policy class, harm taxonomy, confidence, human-review decision,
provenance); (5) provenance as a first-class field — every sample must answer who generated it, from
what seed, with what model revision, under which policy template, using which post-filters, and whether
it contains any real/human-authored content.

### Safety, ethical, and legal risks

The basic problem is not that synthetic data is fake — it is that it can be *falsely reassuring*.
Main risks: misinformation / unrealistic safety behavior; privacy and consent (lawful basis, participant
rights); copyright and data-rights uncertainty; impersonation / digital-replica / deceptive-deployment
harms; and dual-use / jailbreak contamination (over-fitting a benchmark rather than the real threat
model). Mitigations worth implementing: layered filtering **plus** human review (never release samples
that only pass an LLM judge); red-teaming before and after generation; privacy-preserving generation
(DP) where private data enters the loop, claimed only when the mechanism and budget are stated;
provenance metadata and watermarking (C2PA, SynthID Text) as complementary controls; and license-safe
source preference (e.g. Common Pile) over undocumented web mixtures.

### Evaluation, validation, and a recommended pipeline

Evaluate along five axes at once: quality, realism, safety, fairness, downstream utility. Pipeline:
sample → schema validation → duplicate/contamination checks → safety & bias screening → human audit on
a stratified sample → realism comparison to a held-out real corpus → downstream training experiment →
safety benchmark suite → release gate.

Suggested metrics: MAUVE for *relative* distributional realism; dialogue-act distribution comparison for
pragmatic realism; RealToxicityPrompts / ToxiGen for toxicity; BOLD / HolisticBias / BBQ for bias;
XSTest (exaggerated refusals) and SafeDialBench (multi-turn jailbreak robustness) for safety calibration;
and a real downstream-utility delta on held-out real data. Suggested starting release gates: structural
validity ≥ 99.5%; near-duplicate rate ≤ 5%; human realism ≥ 4.0/5; MAUVE used comparatively; no large
dialogue-act mode collapse; monotonic toxicity improvement vs baseline; no bias regression vs real
anchor; minimized unsafe compliance without large safe over-refusal; SafeDialBench improvement across
sub-abilities; positive, stable downstream delta.

Suggested first experiments: a model-family ablation (Pythia 410M/1B/1.4B/2.8B + Claude Haiku as a
quality ceiling); a scaffolding ablation (no persona → persona → persona+hidden goals →
persona+goals+dialogue-act plan); a synthetic/real mixture sweep (100/0 … 0/100 on one held-out
benchmark); a judge-calibration study (human vs LLM-judge by risk category); and a privacy/provenance
intervention study (with/without watermark, provenance, DP; measure utility loss and overhead).

### Reproducibility, documentation, and tooling

Document the full *generation condition*, not just the final corpus: model family, exact revision,
system prompt, seed-template version, decoding settings, filter versions, evaluation-suite versions, and
audit procedure. For hosted models, compensate for weaker model reproducibility with stronger run
metadata (save every prompt template, deployment id, API settings, timestamp; treat provider model
changes as experimental drift). Recommended tooling: vLLM (open-model serving), HF Transformers/Datasets,
Distilabel + Argilla (synthetic-data pipelines + curation), NeMo Curator (GPU dedup/cleaning),
lm-evaluation-harness + Inspect (evaluation), DVC (experiment/data versioning), Croissant + C2PA
(provenance), SynthID Text (watermarking), OpenDP (DP — only claim it when the mechanism and budget are
stated).

### Open questions and limitations

"QLVN" could not be identified from a primary-source model card or paper and should be disambiguated
before any model-specific claim. Two broader limitations regardless of model choice: copyright/licensing
norms around generative-AI training remain unsettled, and synthetic conversations still differ
measurably from human dialogue (dialogue-act distributions, multi-turn safety behavior), so purely
synthetic evaluation remains risky without a real-data anchor. High-confidence recommendation: treat
synthetic conversations as an *amplifier* for safety research, not a substitute for documented real
data, human review, or reproducible methodology.
