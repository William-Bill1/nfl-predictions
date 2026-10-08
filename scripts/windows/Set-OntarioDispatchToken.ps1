<#
.SYNOPSIS
    Stores (or removes) the GitHub token the Ontario dispatcher uses, in
    Windows Credential Manager for the current user.

.DESCRIPTION
    Prompts for the token without echoing it, and writes it as a generic
    credential named "nfl-predictions/ontario-dispatch" (protected by Windows
    for this user account). The token is never printed, logged, written to a
    file, or passed on a command line, and nothing is sent to GitHub.

    Use a fine-grained personal access token limited to
    William-Bill1/nfl-predictions with:
      - Actions: Read and write   (dispatch the workflow, read its runs)
      - Contents: Read-only       (read committed captures on main)
    See docs/ONTARIO_DISPATCH_WINDOWS.md.

.PARAMETER Remove
    Delete the stored credential instead.

.EXAMPLE
    .\scripts\windows\Set-OntarioDispatchToken.ps1
.EXAMPLE
    .\scripts\windows\Set-OntarioDispatchToken.ps1 -Remove
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Remove,
    [string]$Target = 'nfl-predictions/ontario-dispatch'
)
$ErrorActionPreference = 'Stop'

if (-not ('OntarioDispatch.CredStore' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace OntarioDispatch {
    public static class CredStore {
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct CREDENTIAL {
            public int Flags; public int Type; public string TargetName; public string Comment;
            public long LastWritten; public int CredentialBlobSize; public IntPtr CredentialBlob;
            public int Persist; public int AttributeCount; public IntPtr Attributes;
            public string TargetAlias; public string UserName;
        }
        [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern bool CredWriteW(ref CREDENTIAL credential, int flags);
        [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        static extern bool CredDeleteW(string target, int type, int flags);

        public static void Write(string target, IntPtr secret, int byteLength) {
            var c = new CREDENTIAL {
                Type = 1,                    // CRED_TYPE_GENERIC
                TargetName = target,
                Comment = "GitHub token for the Ontario spread dispatcher",
                CredentialBlobSize = byteLength,
                CredentialBlob = secret,
                Persist = 2,                 // CRED_PERSIST_LOCAL_MACHINE (this user, this PC)
                UserName = "github-token"
            };
            if (!CredWriteW(ref c, 0)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        }
        public static bool Delete(string target) {
            return CredDeleteW(target, 1, 0);
        }
    }
}
'@
}

if ($Remove) {
    if ($PSCmdlet.ShouldProcess($Target, 'Delete Windows Credential Manager entry')) {
        if ([OntarioDispatch.CredStore]::Delete($Target)) { "Removed credential '$Target'." }
        else { "No credential '$Target' was stored; nothing removed." }
    }
    return
}

$secure = Read-Host -AsSecureString -Prompt "Paste the GitHub fine-grained token for $Target (input hidden)"
if ($secure.Length -eq 0) { throw 'No token entered; nothing stored.' }
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $length = $secure.Length
    # Shape check on the first characters only, read straight from the
    # protected buffer (the full token never becomes a managed string).
    $chars = for ($i = 0; $i -lt [Math]::Min(11, $length); $i++) {
        [char][Runtime.InteropServices.Marshal]::ReadInt16($bstr, $i * 2)
    }
    $prefix = -join $chars
    $chars = $null
    if (-not ($prefix.StartsWith('github_pat_') -or $prefix.StartsWith('ghp_'))) {
        Write-Warning 'This does not look like a GitHub personal access token (github_pat_... or ghp_...).'
    }
    $prefix = $null
    if ($PSCmdlet.ShouldProcess($Target, 'Store token in Windows Credential Manager')) {
        # The BSTR is UTF-16; store exactly its bytes (the dispatcher reads UTF-16).
        [OntarioDispatch.CredStore]::Write($Target, $bstr, $length * 2)
        "Stored the token as '$Target' for $env:USERDOMAIN\$env:USERNAME (value not shown)."
        'Check it with: python scripts\ontario_dispatch.py check'
    }
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    $secure.Dispose()
}
