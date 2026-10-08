<#
.SYNOPSIS
    Removes the Ontario dispatcher's scheduled tasks. Safe to run again.

.DESCRIPTION
    Unregisters the three tasks in "\NFL Predictions\" if present, and the
    folder if it is then empty. The stored GitHub token and the local logs are
    kept unless -RemoveToken / -RemoveLogs is given. GitHub's own cron in the
    workflow is not affected.

.EXAMPLE
    .\scripts\windows\Uninstall-OntarioDispatch.ps1
.EXAMPLE
    .\scripts\windows\Uninstall-OntarioDispatch.ps1 -RemoveToken -RemoveLogs
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$RemoveToken,
    [switch]$RemoveLogs
)
$ErrorActionPreference = 'Stop'
$Folder = '\NFL Predictions\'
$Names = @('Ontario Spread Dispatch', 'Ontario Spread Dispatch Status', 'Ontario Spread Dispatch Recovery')

foreach ($name in $Names) {
    $task = Get-ScheduledTask -TaskPath $Folder -TaskName $name -ErrorAction SilentlyContinue
    if ($task) {
        if ($PSCmdlet.ShouldProcess("$Folder$name", 'Unregister scheduled task')) {
            Unregister-ScheduledTask -TaskPath $Folder -TaskName $name -Confirm:$false
            "Removed $Folder$name"
        }
    }
    else { "Not registered: $Folder$name" }
}

# Remove the folder only if nothing else is in it.
$svc = New-Object -ComObject 'Schedule.Service'
$svc.Connect()
try {
    $f = $svc.GetFolder($Folder.TrimEnd('\'))
    if ($f.GetTasks(1).Count -eq 0 -and $f.GetFolders(0).Count -eq 0) {
        if ($PSCmdlet.ShouldProcess($Folder, 'Delete empty task folder')) {
            $svc.GetFolder('\').DeleteFolder($Folder.Trim('\'), 0)
            "Removed empty folder $Folder"
        }
    }
}
catch { "Folder $Folder not present." }

if ($RemoveToken) {
    & (Join-Path $PSScriptRoot 'Set-OntarioDispatchToken.ps1') -Remove -WhatIf:$WhatIfPreference
}
if ($RemoveLogs) {
    $dir = Join-Path $env:USERPROFILE '.nfl-predictions\ontario-dispatch'
    if ((Test-Path $dir) -and $PSCmdlet.ShouldProcess($dir, 'Delete dispatcher logs and ledger')) {
        Remove-Item -Recurse -Force $dir
        "Removed $dir"
    }
}
