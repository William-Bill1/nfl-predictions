<#
.SYNOPSIS
    Shared helpers for the Ontario dispatcher scripts (dot-sourced). Read-only.

.DESCRIPTION
    Get-BatchLogonRight reports whether an account may "Log on as a batch job"
    (SeBatchLogonRight), which tasks that run "whether the user is logged on or
    not" need. Membership of Administrators doesn't guarantee it: local or
    domain policy can remove the right or deny it (SeDenyBatchLogonRight, which
    wins). The effective local policy can only be read from an elevated
    session; otherwise the result is 'Unverified', never assumed.
#>

# Logon-type SIDs that a task's batch logon doesn't carry, and the one it does.
$script:InteractiveOnlySids = @('S-1-5-4', 'S-1-5-14')        # INTERACTIVE, REMOTE INTERACTIVE
$script:BatchSid = 'S-1-5-3'                                   # BATCH

function ConvertTo-RightSid([string]$Entry) {
    # secedit lists SIDs as *S-1-...; plain names are translated when possible.
    if ($Entry.StartsWith('*')) { return $Entry.Substring(1) }
    try { return (New-Object Security.Principal.NTAccount $Entry).Translate([Security.Principal.SecurityIdentifier]).Value }
    catch { return $Entry }
}

function Test-BatchLogonRightFromPolicy {
    <# Pure decision from exported policy lines and the account's SIDs (testable). #>
    param([string[]]$PolicyLines, [string[]]$Sids)
    $rights = @{ SeBatchLogonRight = @(); SeDenyBatchLogonRight = @() }
    foreach ($line in $PolicyLines) {
        if ($line -match '^\s*(SeBatchLogonRight|SeDenyBatchLogonRight)\s*=\s*(.*)$') {
            $rights[$Matches[1]] = @($Matches[2] -split ',' | ForEach-Object { $_.Trim() } |
                Where-Object { $_ } | ForEach-Object { ConvertTo-RightSid $_ })
        }
    }
    $denied = @($rights.SeDenyBatchLogonRight | Where-Object { $Sids -contains $_ })
    if ($denied.Count) {
        return [pscustomobject]@{ Status = 'Denied'; Detail = "denied by policy via $($denied -join ', ')" }
    }
    $granted = @($rights.SeBatchLogonRight | Where-Object { $Sids -contains $_ })
    if ($granted.Count) {
        return [pscustomobject]@{ Status = 'Granted'; Detail = "granted via $($granted -join ', ')" }
    }
    return [pscustomobject]@{ Status = 'NotGranted'; Detail = 'neither the account nor any of its groups holds SeBatchLogonRight' }
}

function Get-BatchLogonRight {
    <# Effective local policy for the current account (needs elevation to read). #>
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $elevated = ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $elevated) {
        return [pscustomobject]@{ Status = 'Unverified'; Detail = 'reading the effective user rights needs an elevated session' }
    }
    $cfg = Join-Path ([IO.Path]::GetTempPath()) ("ontario-dispatch-rights-{0}.inf" -f [guid]::NewGuid())
    try {
        $null = & secedit.exe /export /areas USER_RIGHTS /cfg $cfg /quiet
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $cfg)) {
            return [pscustomobject]@{ Status = 'Unverified'; Detail = "secedit export failed (exit $LASTEXITCODE)" }
        }
        $lines = Get-Content $cfg
    }
    finally { Remove-Item $cfg -Force -ErrorAction SilentlyContinue }
    # The account and its groups as a batch logon would see them.
    $sids = @($id.User.Value) + @($id.Groups | ForEach-Object { $_.Value } |
        Where-Object { $script:InteractiveOnlySids -notcontains $_ -and $_ -notlike 'S-1-5-5-*' }) + $script:BatchSid
    return Test-BatchLogonRightFromPolicy -PolicyLines $lines -Sids $sids
}
