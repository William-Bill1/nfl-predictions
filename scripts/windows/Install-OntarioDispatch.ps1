<#
.SYNOPSIS
    Registers (or updates) the three Task Scheduler tasks that dispatch the
    Ontario Spread Capture GitHub workflow. Safe to run again.

.DESCRIPTION
    Tasks, in the "\NFL Predictions\" folder:
      Ontario Spread Dispatch           Wed 12:05 + 12:30, Sun 09:05 + 09:30
      Ontario Spread Dispatch Status    Wed 13:00, Sun 10:00
      Ontario Spread Dispatch Recovery  at startup (+3 min) and on resume from sleep (+2 min)
    Times are local, so the computer's timezone must follow America/Toronto
    (Windows "Eastern Standard Time" with automatic DST). This is verified, not
    assumed.

    The tasks run as you, "whether logged on or not" (logon type Password), so
    they work after an unattended Windows update reboot and can read your token
    from Windows Credential Manager. Task Scheduler asks for and stores your
    Windows password; this script never prints or keeps it.

    Task settings come from scripts\ontario_dispatch.py task-xml (tested):
    wake to run, start when available, network required, run on battery, one
    instance at a time, time limits, restart on failure.

.PARAMETER PlanOnly
    Run every check and print the task definitions; register nothing.

.EXAMPLE
    .\scripts\windows\Install-OntarioDispatch.ps1 -PlanOnly
.EXAMPLE
    .\scripts\windows\Install-OntarioDispatch.ps1        # from an elevated PowerShell
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RepoRoot,
    [string]$Python,
    [ValidateSet('credman', 'git')][string]$Auth = 'credman',
    [switch]$PlanOnly,
    [switch]$AllowStorePython
)
$ErrorActionPreference = 'Stop'
# Resolved here, not as a parameter default: Windows PowerShell 5.1 can leave
# $PSScriptRoot empty while binding parameters.
$Here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $Here '..\..')).Path }
. (Join-Path $Here 'OntarioDispatch.Common.ps1')
$Folder = '\NFL Predictions\'
$Tasks = [ordered]@{
    dispatch = 'Ontario Spread Dispatch'
    status   = 'Ontario Spread Dispatch Status'
    recovery = 'Ontario Spread Dispatch Recovery'
}
if (-not $Python) { $Python = Join-Path $RepoRoot 'venv-dispatch\Scripts\python.exe' }
$Dispatcher = Join-Path $RepoRoot 'scripts\ontario_dispatch.py'
$User = "$env:USERDOMAIN\$env:USERNAME"

# 1. Timezone must follow America/Toronto (Eastern, DST on).
$tz = Get-TimeZone
$dstOff = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\TimeZoneInformation').DynamicDaylightTimeDisabled
if ($tz.Id -ne 'Eastern Standard Time' -or -not $tz.SupportsDaylightSavingTime -or $dstOff -eq 1) {
    throw "Timezone is '$($tz.Id)' (DST adjustment disabled: $($dstOff -eq 1)). Task times are local, so set Windows to (UTC-05:00) Eastern Time (US & Canada) with 'Adjust for daylight saving time automatically' on, then run this again."
}
"Timezone: $($tz.Id), DST automatic - OK"

# 2. Files.
foreach ($p in @($Python, $Dispatcher)) {
    if (-not (Test-Path $p)) { throw "Not found: $p" }
}

# 2b. The interpreter that will actually run, asked of Python itself: for a
# venv, the base install it was created from. Microsoft Store Python (an MSIX
# app started through an execution alias) isn't reliable from a task that runs
# while you are signed out, and it redirects AppData writes, so a python.org
# CPython 3.12-3.14 with requirements-dispatch.txt is required.
$runtimeOut = & $Python $Dispatcher runtime
try { $rt = ($runtimeOut -join "`n") | ConvertFrom-Json }
catch { throw "Could not run '$Python $Dispatcher runtime': $($runtimeOut -join ' ')" }
"Python: $($rt.implementation) $($rt.version) ($($rt.bits)-bit) at $($rt.executable)"
"  base interpreter: $($rt.base_executable_resolved)"
"  dependencies: $(($rt.dependencies.PSObject.Properties | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ', ')"
$other = @($rt.problems | Where-Object { $_ -notlike '*Store Python*' })
if ($other.Count) {
    $msg = "The dispatcher's Python isn't usable: $($other -join '; '). Use python.org CPython 3.12-3.14 and install requirements-dispatch.txt (see docs\ONTARIO_DISPATCH_WINDOWS.md)."
    if ($PlanOnly) { Write-Warning $msg } else { throw $msg }
}
if ($rt.store_python) {
    $msg = "Python at '$Python' is the Microsoft Store Python or a venv made from it (base: $($rt.base_executable_resolved)). Scheduled tasks that run while you are signed out may fail to start it. Install Python from python.org, create a venv for the dispatcher (see docs\ONTARIO_DISPATCH_WINDOWS.md) and pass it with -Python, or use -AllowStorePython to try anyway."
    if ($AllowStorePython -or $PlanOnly) { Write-Warning $msg } else { throw $msg }
}

# 3. The dispatcher's own checks: OS rules vs America/Toronto across DST, and the token's presence.
& $Python $Dispatcher check --auth $Auth
$check = $LASTEXITCODE
if ($check -eq 4) { throw 'The dispatcher check failed: the computer timezone does not follow America/Toronto, or the Python runtime is incomplete (see above).' }
if ($check -eq 2) {
    $msg = "No usable GitHub token ($Auth). Run scripts\windows\Set-OntarioDispatchToken.ps1 first."
    if ($PlanOnly) { Write-Warning $msg } else { throw $msg }
}

# 4. Task definitions (generated and tested in Python).
$xml = [ordered]@{}
foreach ($key in $Tasks.Keys) {
    $xml[$key] = (& $Python $Dispatcher task-xml --task $key --python $Python --repo-root $RepoRoot --user $User) -join "`r`n"
    if ($LASTEXITCODE -ne 0 -or -not $xml[$key]) { throw "Could not generate the '$key' task definition." }
}

# 5. "Log on as a batch job" for this account, from the effective local policy
# (Administrators membership alone doesn't guarantee it). Needs elevation.
$batch = Get-BatchLogonRight
"Log on as a batch job ($User): $($batch.Status) - $($batch.Detail)"

if ($PlanOnly) {
    if ($batch.Status -ne 'Granted') { Write-Warning "Log on as a batch job is $($batch.Status); the install (elevated) refuses unless it is Granted." }
    foreach ($key in $Tasks.Keys) {
        "`n===== $Folder$($Tasks[$key]) ====="
        $xml[$key]
    }
    "`nPlan only: nothing registered."
    return
}

# 6. Registering a stored-password task in a new folder needs an elevated session.
$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elevated) {
    throw 'Run this from an elevated PowerShell (Run as administrator). The tasks still run as you, with least privilege.'
}
if ($batch.Status -ne 'Granted') {
    throw "This account can't be confirmed to have 'Log on as a batch job' ($($batch.Status): $($batch.Detail)). Tasks that run while you are signed out would fail to start. Grant it in Local Security Policy (secpol.msc > Local Policies > User Rights Assignment), or ask your administrator if it is set by domain policy."
}

# 7. Register (or replace) each task. Task Scheduler stores the password; it is cleared here afterwards.
$cred = Get-Credential -UserName $User -Message "Windows password for $User (Task Scheduler stores it so the tasks run after a reboot while you are signed out). For a Microsoft account, use the account password, not the PIN."
$plain = $cred.GetNetworkCredential().Password
try {
    foreach ($key in $Tasks.Keys) {
        if ($PSCmdlet.ShouldProcess("$Folder$($Tasks[$key])", 'Register scheduled task')) {
            Register-ScheduledTask -Xml $xml[$key] -TaskName $Tasks[$key] -TaskPath $Folder `
                -User $User -Password $plain -Force | Out-Null
            "Registered $Folder$($Tasks[$key])"
        }
    }
}
finally {
    $plain = $null
    $cred = $null
}

"`nInstalled. Inspect with: .\scripts\windows\Get-OntarioDispatchStatus.ps1"
"Try a dry run with:     & '$Python' '$Dispatcher' run --mode dispatch --dry-run"
