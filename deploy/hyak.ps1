<#
.SYNOPSIS
  One-command Hyak (klone) driver for the survival audit: assemble locally, push+submit, check
  status, and fetch+verify results. Run from anywhere; paths resolve to the repo root.

.DESCRIPTION
  Klone forbids SSH keys (2FA/Duo on every connection) and Windows OpenSSH cannot multiplex, so each
  remote step prompts Duo. This wraps the whole flow into ONE PowerShell command per action and
  minimizes the remote round-trips. Edit deploy/hyak.config.ps1 once first.

.EXAMPLE
  ./deploy/hyak.ps1 setup     # one-time: build the conda env + prefetch tokenizers on a compute node
  ./deploy/hyak.ps1 push      # assemble corpus+shards locally, upload, submit the survival array
  ./deploy/hyak.ps1 status    # squeue + how many shards have finished
  ./deploy/hyak.ps1 fetch     # download results, aggregate, and verify the counterfactual control
  ./deploy/hyak.ps1 push -DryRun   # print every command without running it
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'push', 'status', 'fetch', 'report', 'cancel', 'help')]
    [string]$Action = 'help',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'hyak.config.ps1')
$c = $HyakConfig

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$sshTarget = "$($c.NetId)@$($c.LoginHost)"
$remoteRoot = $c.RemoteRoot

function Assert-Configured {
    if (($c.NetId + $c.Account + $c.RemoteRoot + $c.EnvPrefix) -match 'REPLACE') {
        Write-Warning 'deploy/hyak.config.ps1 still has REPLACE_ placeholders -- edit it before running remote actions.'
    }
}

function Invoke-Native {
    param([string]$Exe, [string[]]$Argv)
    Write-Host ">> $Exe $($Argv -join ' ')" -ForegroundColor DarkGray
    if ($DryRun) { return }
    & $Exe @Argv
    if ($LASTEXITCODE -ne 0) { throw "$Exe exited with code $LASTEXITCODE" }
}

function Merge-Bytes {
    # Concatenate files byte-for-byte into $Dest with no re-encoding and no BOM (unlike
    # Set-Content -Encoding utf8 on PowerShell 5.1). Relative paths resolve against the repo root.
    param([string[]]$Sources, [string]$Dest)
    $resolve = { param($p) if ([System.IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $root $p } }
    $ms = New-Object System.IO.MemoryStream
    try {
        foreach ($s in $Sources) {
            $bytes = [System.IO.File]::ReadAllBytes((& $resolve $s))
            $ms.Write($bytes, 0, $bytes.Length)
        }
        [System.IO.File]::WriteAllBytes((& $resolve $Dest), $ms.ToArray())
    }
    finally { $ms.Dispose() }
}

function Invoke-Assemble {
    Write-Host '== assemble per-model corpora + manifest + shards (local) ==' -ForegroundColor Cyan
    # Slot-planting positions. `system` is positional (no slot planted); `tool_output` plants the
    # tool slot and is generated only for the agent_tool arm. build-manifest is slot-aware, so
    # tool_output expands only on bases that carry the slot.
    $core = @('prefix', 'middle', 'end', 'old_turn', 'recent_turn')
    $agentPos = $core + @('tool_output')
    Push-Location $root
    try {
        New-Item -ItemType Directory -Force -Path 'data/pilot', 'data/shards' | Out-Null
        # Clear shards ONCE before the loop; each model appends its own <model>_shard_*.jsonl below.
        if (-not $DryRun) {
            Get-ChildItem 'data/shards' -Filter '*.jsonl' -ErrorAction SilentlyContinue | Remove-Item -Force
        }
        $perModelBases = @()
        foreach ($m in $c.Models_Fleet) {
            Write-Host "-- $($m.Id): bind bases to $($m.Tokenizer) at content length $($m.TargetLen) --" -ForegroundColor DarkCyan
            $synth = "data/pilot/synthetic_$($m.Id).jsonl"
            $agent = "data/pilot/agent_tool_$($m.Id).jsonl"
            $longdoc = "data/pilot/longdoc_$($m.Id).jsonl"
            $combined = "data/pilot/base_conversations_$($m.Id).jsonl"
            # Length-match each arm to THIS model's own tokenizer (hf), namespace ids by model so all
            # models' sets coexist collision-free. (a) synthetic multi-turn + long-document chat:
            Invoke-Native $py (@('-m', 'trigger_audit.generation.conversation_generator',
                    '--model-id', $m.Tokenizer, '--tokenizer-backend', 'hf',
                    '--target-length', "$($m.TargetLen)", '--count', "$($c.SynthCount)",
                    '--generation-backend', 'mock', '--chat-format', $m.ChatFormat,
                    '--base-id-namespace', $m.Id, '--positions') + $core + @('--output', $synth))
            # (b) agent/tool bases -- the only arm tool_output plants on (plants the tool slot too).
            # Skipped for a model whose template can't render the `tool` role (AgentTool = $false).
            $doAgent = -not ($m.ContainsKey('AgentTool') -and ($m.AgentTool -eq $false))
            if ($doAgent) {
                Invoke-Native $py (@('-m', 'trigger_audit.generation.conversation_generator',
                        '--model-id', $m.Tokenizer, '--tokenizer-backend', 'hf',
                        '--target-length', "$($m.TargetLen)", '--count', "$($c.AgentToolCount)",
                        '--generation-backend', 'mock', '--chat-format', $m.ChatFormat, '--families', 'agent_tool',
                        '--base-id-namespace', $m.Id, '--positions') + $agentPos + @('--output', $agent))
            }
            else {
                Write-Host "   (agent_tool arm skipped for $($m.Id): template has no tool role)" -ForegroundColor DarkYellow
            }
            # (c) long documents from big.txt:
            Invoke-Native $py (@('-m', 'trigger_audit.io.dataset_adapter',
                    '--source', 'longdoc', '--text-path', $c.TextPath,
                    '--model-id', $m.Tokenizer, '--tokenizer-backend', 'hf',
                    '--target-length', "$($m.TargetLen)", '--limit', "$($c.LongdocCount)",
                    '--chat-format', $m.ChatFormat, '--base-id-namespace', $m.Id,
                    '--positions') + $core + @('--output', $longdoc))
            # (d) real H4 arm: merge any pre-pulled data/real/<source>_<Id>.jsonl that exists (run
            # scripts/pull_real_arm.py once). A model with no real file (e.g. Gemma) simply has none.
            $arms = @($synth, $longdoc)
            if ($doAgent) { $arms += $agent }
            foreach ($src in $c.RealSources) {
                $real = "data/real/${src}_$($m.Id).jsonl"
                if (Test-Path (Join-Path $root $real)) {
                    Write-Host "   + real arm: $real" -ForegroundColor DarkGray
                    $arms += $real
                }
                else {
                    Write-Host "   (no real arm file $real -- skipping)" -ForegroundColor DarkYellow
                }
            }
            if (-not $DryRun) { Merge-Bytes $arms $combined }
            $perModelBases += $combined
            # This model's shards only: override the experiment's model + base file. build-manifest is
            # slot-aware, so tool_output expands only for the agent_tool bases in this combined store.
            Invoke-Native $py @('-m', 'trigger_audit', 'build-manifest', $c.Experiment,
                '--model-id', $m.Id, '--base-conversations', $combined)
        }
        # Merge all per-model base sets into the single combined store the runner + report read.
        if (-not $DryRun) { Merge-Bytes $perModelBases 'data/pilot/base_conversations.jsonl' }
    }
    finally { Pop-Location }
}

function Write-RemoteEnv {
    $lines = @(
        "ACCOUNT=$($c.Account)", "PARTITION=$($c.Partition)", "ENV_PREFIX=$($c.EnvPrefix)",
        "REMOTE_ROOT=$($c.RemoteRoot)", "EMAIL=$($c.Email)",
        "MODELS_CONFIG=$($c.Models)", "POLICIES_CONFIG=$($c.Policies)",
        'BASES=data/pilot/base_conversations.jsonl', 'BACKEND=hf'
    )
    # Gemma's tokenizer is gated: forward YOUR local HF_TOKEN so the setup job's prefetch can pull it
    # (hyak_submit.sh adds it to the sbatch env). Only written when you export HF_TOKEN locally; it is
    # your token going to your own cluster home. Skip this (leave HF_TOKEN unset) and instead run
    # `huggingface-cli login` on klone if you prefer not to place the token in a file.
    if ($env:HF_TOKEN) { $lines += "HF_TOKEN=$($env:HF_TOKEN)" }
    if (-not $DryRun) {
        [System.IO.File]::WriteAllText((Join-Path $root 'deploy/hyak.remote.env'), ($lines -join "`n") + "`n")
    }
    Write-Host 'wrote deploy/hyak.remote.env (LF)' -ForegroundColor DarkGray
}

function New-Payload {
    # Only pack paths that exist (setup runs before any shards are assembled).
    $candidates = @('pyproject.toml', 'README.md', 'src', 'configs', 'deploy', 'scripts',
        'data/shards', 'data/pilot/base_conversations.jsonl', 'data/triggers/triggers.jsonl')
    $items = $candidates | Where-Object { Test-Path (Join-Path $root $_) }
    Push-Location $root
    try {
        if ((Test-Path 'deploy/payload.tgz') -and (-not $DryRun)) { Remove-Item 'deploy/payload.tgz' -Force }
        Invoke-Native 'tar' (@('--exclude=deploy/payload.tgz', '--exclude=deploy/results.tgz',
                '--exclude=*/__pycache__/*', '-czf', 'deploy/payload.tgz') + $items)
    }
    finally { Pop-Location }
}

function Send-And-Submit {
    param([string]$What)   # 'run' or 'setup'
    Push-Location $root
    try {
        Write-Host "== upload payload + submit ($What) -- approve Duo (~2x) ==" -ForegroundColor Cyan
        # Stage the upload in scrubbed storage (always writable, TBs of space), never the 10 GB home
        # dir -- scp to home fails once the home quota/inodes are exhausted. Then move it into place.
        $stage = "/gscratch/scrubbed/$($c.NetId)_ta_payload.tgz"
        Invoke-Native 'scp' @('deploy/payload.tgz', "${sshTarget}:$stage")
        $remote = "mkdir -p '$remoteRoot' && mv '$stage' '$remoteRoot/payload.tgz' && cd '$remoteRoot' && tar xzf payload.tgz && bash deploy/hyak_submit.sh $What"
        Invoke-Native 'ssh' @($sshTarget, $remote)
    }
    finally { Pop-Location }
}

function Invoke-Report {
    Push-Location $root
    try {
        if (-not (Test-Path 'outputs/survival_results')) {
            Write-Warning 'no outputs/survival_results -- run `fetch` first.'; return
        }
        Invoke-Native $py @('-m', 'trigger_audit', 'score-survival', 'outputs/survival_results')
        if (-not $DryRun) {
            $parts = Get-ChildItem 'outputs/survival_results' -Filter '*.jsonl' | Sort-Object Name
            Merge-Bytes ($parts | ForEach-Object { $_.FullName }) 'outputs/survival.jsonl'
        }
        Invoke-Native $py @('scripts/pilot_report.py', 'outputs/survival.jsonl', 'data/pilot/base_conversations.jsonl')
    }
    finally { Pop-Location }
}

switch ($Action) {
    'setup' {
        Assert-Configured; Write-RemoteEnv; New-Payload; Send-And-Submit 'setup'
        Write-Host 'Setup job submitted. Check it with:  ./deploy/hyak.ps1 status' -ForegroundColor Green
    }
    'push' {
        Assert-Configured; Invoke-Assemble; Write-RemoteEnv; New-Payload; Send-And-Submit 'run'
        Write-Host 'Survival array submitted. Watch it with:  ./deploy/hyak.ps1 status' -ForegroundColor Green
    }
    'status' {
        $remote = "echo '== your jobs =='; squeue -u '$($c.NetId)' -o '%.18i %.12P %.20j %.8T %.10M %R'; " +
        "echo '== finished / total shards =='; " +
        "printf '%s / %s\n' " +
        "`"`$(ls '$remoteRoot'/outputs/survival_results/*.jsonl 2>/dev/null | wc -l)`" " +
        "`"`$(ls '$remoteRoot'/data/shards/*.jsonl 2>/dev/null | wc -l)`""
        Invoke-Native 'ssh' @($sshTarget, $remote)
    }
    'fetch' {
        Write-Host '== pack + download results, then verify locally -- approve Duo (~2x) ==' -ForegroundColor Cyan
        Invoke-Native 'ssh' @($sshTarget, "cd '$remoteRoot' && tar czf results.tgz outputs/survival_results outputs/logs outputs/final_prompts 2>/dev/null && echo packed")
        Push-Location $root
        try {
            if ((Test-Path 'deploy/results.tgz') -and (-not $DryRun)) { Remove-Item 'deploy/results.tgz' -Force }
            Invoke-Native 'scp' @("${sshTarget}:$remoteRoot/results.tgz", 'deploy/results.tgz')
            if (-not $DryRun) {
                # Clear stale local results so the aggregate reflects only what came back this fetch.
                if (Test-Path 'outputs/survival_results') {
                    Get-ChildItem 'outputs/survival_results' -Filter '*.jsonl' | Remove-Item -Force
                }
                Invoke-Native 'tar' @('-xzf', 'deploy/results.tgz', '-C', '.')
            }
        }
        finally { Pop-Location }
        Invoke-Report
    }
    'report' { Invoke-Report }
    'cancel' { Invoke-Native 'ssh' @($sshTarget, "scancel -u '$($c.NetId)' --name=trigaudit-survival") }
    default {
        Write-Host @'
Hyak driver -- usage:
  ./deploy/hyak.ps1 setup    one-time: build conda env + prefetch tokenizers (compute-node job)
  ./deploy/hyak.ps1 push     assemble corpus+shards locally, upload, submit the survival array
  ./deploy/hyak.ps1 status   squeue + finished/total shard counts
  ./deploy/hyak.ps1 fetch    download results, aggregate, verify the counterfactual control
  ./deploy/hyak.ps1 report   re-run the local aggregation on already-fetched results (no Duo)
  ./deploy/hyak.ps1 cancel   scancel the survival array
Add -DryRun to print commands without executing. Edit deploy/hyak.config.ps1 first.
'@
    }
}
