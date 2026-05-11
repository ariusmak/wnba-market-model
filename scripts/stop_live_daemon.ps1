param(
    [string]$LockPath = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $LockPath) {
    $LockPath = Join-Path $RepoRoot "data\runs\live_daemon\live_daemon.lock"
}

if (-not (Test-Path $LockPath)) {
    Write-Host "No live daemon lock found at $LockPath"
    $EscapedRepo = [regex]::Escape($RepoRoot)
    try {
        $Candidates = Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "python" -and
                $_.CommandLine -match "15_live_daemon.py" -and
                $_.CommandLine -match $EscapedRepo
            }
    } catch {
        Write-Warning "Could not inspect process command lines for no-lock fallback: $($_.Exception.Message)"
        exit 1
    }
    if (-not $Candidates) {
        exit 0
    }
    foreach ($Candidate in $Candidates) {
        Write-Host "Stopping WNBA live daemon pid $($Candidate.ProcessId) without lock"
        Stop-Process -Id ([int]$Candidate.ProcessId) -Force
    }
    Write-Host "Stopped."
    exit 0
}

$Lock = Get-Content $LockPath -Raw | ConvertFrom-Json
$PidToStop = [int]$Lock.pid
if (-not $PidToStop) {
    throw "Lock file does not contain a valid pid: $LockPath"
}

$Process = Get-Process -Id $PidToStop -ErrorAction SilentlyContinue
if (-not $Process) {
    Write-Host "No process found for pid $PidToStop; removing stale lock."
    Remove-Item -LiteralPath $LockPath -Force
    exit 0
}

Write-Host "Stopping WNBA live daemon pid $PidToStop"
Stop-Process -Id $PidToStop -Force
Start-Sleep -Seconds 1
if (Test-Path $LockPath) {
    Remove-Item -LiteralPath $LockPath -Force
}
Write-Host "Stopped."
