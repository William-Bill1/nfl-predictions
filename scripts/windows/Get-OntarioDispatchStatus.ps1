<#
.SYNOPSIS
    Read-only view of the Ontario dispatcher on this computer: tasks and their
    settings, last and next runs, timezone, token presence, wake timers and the
    latest log lines. Changes nothing.

.EXAMPLE
    .\scripts\windows\Get-OntarioDispatchStatus.ps1
#>
[CmdletBinding()]
param(
    [string]$RepoRoot,
    [string]$Python,
    [int]$LogLines = 20
)
$ErrorActionPreference = 'Continue'
$Folder = '\NFL Predictions\'
$Target = 'nfl-predictions/ontario-dispatch'
$LogFile = Join-Path $env:USERPROFILE '.nfl-predictions\ontario-dispatch\dispatch.log'
$Here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $Here '..\..')).Path }
. (Join-Path $Here 'OntarioDispatch.Common.ps1')

'== Tasks'
$tasks = Get-ScheduledTask -TaskPath $Folder -ErrorAction SilentlyContinue
# The interpreter the tasks actually run, unless one is given.
if (-not $Python -and $tasks) { $Python = @($tasks)[0].Actions[0].Execute }
if (-not $Python) { $Python = Join-Path $RepoRoot 'venv-dispatch\Scripts\python.exe' }
if (-not $tasks) { '  none registered (run Install-OntarioDispatch.ps1)' }
foreach ($t in $tasks) {
    $info = Get-ScheduledTaskInfo -TaskName $t.TaskName -TaskPath $t.TaskPath
    $s = $t.Settings
    "  $($t.TaskName): $($t.State)"
    "    logon=$($t.Principal.LogonType) user=$($t.Principal.UserId) runlevel=$($t.Principal.RunLevel)"
    "    wake=$($s.WakeToRun) startWhenAvailable=$($s.StartWhenAvailable) networkRequired=$($s.RunOnlyIfNetworkAvailable) batteryStart=$(-not $s.DisallowStartIfOnBatteries) batteryStop=$($s.StopIfGoingOnBatteries) instances=$($s.MultipleInstances) limit=$($s.ExecutionTimeLimit)"
    "    last run $($info.LastRunTime) result 0x$('{0:X}' -f $info.LastTaskResult)  next run $($info.NextRunTime)"
}

'== Timezone'
$tz = Get-TimeZone
"  $($tz.Id); DST supported=$($tz.SupportsDaylightSavingTime); now $(Get-Date -Format 'yyyy-MM-dd HH:mm zzz')"

'== Token (presence only)'
$found = cmdkey /list | Select-String -SimpleMatch $Target
if ($found) { "  stored as '$Target' (value not shown)" } else { "  not stored (run Set-OntarioDispatchToken.ps1)" }

'== Wake timers (active power plan)'
powercfg /q SCHEME_CURRENT SUB_SLEEP RTCWAKE 2>$null |
    Select-String -Pattern 'Current (AC|DC) Power Setting Index' |
    ForEach-Object { '  ' + $_.ToString().Trim() + '   (0=disabled, 1=enabled, 2=important only)' }

'== Log on as a batch job (needed while signed out)'
$batch = Get-BatchLogonRight
"  $($batch.Status) - $($batch.Detail)"

"== Python runtime ($Python)"
if (Test-Path $Python) {
    $rt = (& $Python (Join-Path $RepoRoot 'scripts\ontario_dispatch.py') runtime) -join "`n" | ConvertFrom-Json
    "  $($rt.implementation) $($rt.version), base $($rt.base_executable_resolved)"
    if ($rt.ok) { '  OK' } else { $rt.problems | ForEach-Object { "  PROBLEM: $_" } }
    '== Dispatcher check'
    & $Python (Join-Path $RepoRoot 'scripts\ontario_dispatch.py') check | ForEach-Object { "  $_" }
}
else { "  Python not found: $Python" }

'== Unattended evidence (latest auth_ok lines; session 0 = ran with no desktop)'
if (Test-Path $LogFile) {
    Get-Content $LogFile | Select-String -SimpleMatch '"outcome": "auth_ok"' | Select-Object -Last 3 |
        ForEach-Object { "  $_" }
}

"== Last $LogLines log lines ($LogFile)"
if (Test-Path $LogFile) { Get-Content $LogFile -Tail $LogLines | ForEach-Object { "  $_" } }
else { '  no log yet' }
