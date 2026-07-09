# Edit these once for your Hyak account, then drive everything with deploy/hyak.ps1.
# Nothing else in the deploy kit needs hand-editing -- these values flow to the cluster on push.
# Find your account with `hyakalloc` on klone; put storage on /gscratch (home is only ~10 GB).

$HyakConfig = @{
    NetId      = 'sks0417'                                  # your UW NetID (login user)
    LoginHost  = 'klone.hyak.uw.edu'
    Account    = 'stf'                                      # Slurm account (from `hyakalloc`; NOT your netid)
    Partition  = 'ckpt-all'                                 # free, preemptible CPU partition
    # PROD (persistent) storage. Item 1 (scorer/metadata fix + pilot re-verify) is done, so we have
    # switched off scrubbed (free but auto-purged after 21 days -- its purge can corrupt a conda env
    # over a long run) to group storage /gscratch/stf/... which persists. This is a NEW path with no
    # env yet, so re-run `setup` once before `push` (see the note in `docs/ONE_SHOT_PLAN.md` sect.4).
    # Uploads still stage via scrubbed either way.
    RemoteRoot = '/gscratch/stf/sks0417/trigger_audit'
    EnvPrefix  = '/gscratch/stf/sks0417/ta_env'
    Email      = 'sks0417@uw.edu'

    # What the cluster runs (assembled locally + pushed by `push`). WAVE 2 is active: the full
    # delivery audit -- 5 policies x 7 positions (incl. system + tool_output) x 3 budgets x 5 triggers
    # x counterfactual, across FOUR tokenizers (adds Gemma: system-merge role_migration + strict-
    # alternation template_incompatible) and FOUR data arms (synthetic + long-doc + agent_tool +
    # the real LMSYS/WildChat H4 arm merged from data/real/). build-manifest is slot-aware, so
    # tool_output only expands on agent_tool bases. To revert to the pilot smoke, point these three
    # at configs/pilot/*.pilot*.yaml (and drop TargetLen to 256 + SynthCount/LongdocCount below).
    Experiment = 'configs/prod/experiment.prod.yaml'
    Models     = 'configs/prod/models.prod.yaml'
    Policies   = 'configs/prod/policies.prod.yaml'

    # The assembly FLEET: one entry per model. Bases are length-matched to EACH model's own tokenizer
    # (--tokenizer-backend hf) and tagged with its Id so per-model sets coexist in one combined store.
    # TargetLen is the base CONTENT length -- keep it ABOVE the experiment's largest context_length
    # budget so truncation actually binds. Budgets are [512,1024,2048]: Qwen/Gemma TargetLen=4096 (all
    # three budgets truncate); Pythia/TinyLlama TargetLen=1536 (below their 2048 window, so the 2048
    # budget acts as a clean no-truncation control). Gemma is GATED (needs a gated-read HF_TOKEN in
    # setup for the tokenizer prefetch); its real LMSYS/WildChat arm is intentionally omitted (its
    # value is H2 template divergence, exercised by the generated arms).
    # AgentTool = $false skips the agent/tool arm for a model whose chat template cannot render the
    # `tool` role (Gemma rejects it outright -- its value here is H2 chat template divergence, not
    # tools). Omitted = generated. Qwen/TinyLlama/Pythia all render the tool role.
    Models_Fleet = @(
        @{ Id = 'qwen3-0_6b';          Tokenizer = 'Qwen/Qwen3-0.6B';                    ChatFormat = 'chat'; TargetLen = 4096 }
        @{ Id = 'pythia-1b';           Tokenizer = 'EleutherAI/pythia-1b';               ChatFormat = 'base'; TargetLen = 1536 }
        @{ Id = 'tinyllama-1_1b-chat'; Tokenizer = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'; ChatFormat = 'chat'; TargetLen = 1536 }
        @{ Id = 'gemma-3-1b-it';       Tokenizer = 'google/gemma-3-1b-it';               ChatFormat = 'chat'; TargetLen = 4096; AgentTool = $false }
    )

    # Per-model corpus sizes. SynthCount = multi-turn+long-doc chat seeds; AgentToolCount = agent/tool
    # bases (the only ones tool_output plants on); LongdocCount = big.txt slices. The real LMSYS/
    # WildChat arm is pre-pulled to data/real/{lmsys,wildchat}_<model>.jsonl and merged as-is (run
    # scripts/pull_real_arm.py once; models without a data/real file simply get no real arm).
    SynthCount     = 60
    AgentToolCount = 20
    LongdocCount   = 15
    TextPath       = 'big.txt'
    # Real-arm sources merged from data/real/ when a matching per-model file exists.
    RealSources    = @('lmsys', 'wildchat')
}
