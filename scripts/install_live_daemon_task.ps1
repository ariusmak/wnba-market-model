param(
    [string]$TaskName = "WNBA Live Daemon",
    [string]$PythonExe = "",
    [int]$Year = 2026,
    [int]$MarketPollSeconds = 180,
    [int]$WorkerCheckSeconds = 300
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) {
    $cmd = Get-Command python -ErrorAction Stop
    $PythonExe = $cmd.Source
}

$DaemonScript = Join-Path $RepoRoot "pipelines\07_live\15_live_daemon.py"
if (-not (Test-Path $DaemonScript)) {
    throw "Missing daemon script: $DaemonScript"
}

$Arguments = @(
    "`"$DaemonScript`"",
    "--year", $Year,
    "--market-poll-s", $MarketPollSeconds,
    "--worker-check-s", $WorkerCheckSeconds
) -join " "

$Registered = $false
try {
    $Action = New-ScheduledTaskAction -Execute $PythonExe -Argument $Arguments -WorkingDirectory $RepoRoot
    $Trigger = New-ScheduledTaskTrigger -AtLogOn
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "Runs the WNBA production live daemon for Kalshi market monitoring and due data refreshes." `
        -Force | Out-Null
    $Registered = $true
} catch {
    Write-Warning "Register-ScheduledTask failed: $($_.Exception.Message)"
    Write-Warning "Falling back to schtasks.exe for current-user logon registration."

    $TaskRun = "`"$PythonExe`" $Arguments"
    $SchTasksArgs = @(
        "/Create",
        "/TN", $TaskName,
        "/TR", $TaskRun,
        "/SC", "ONLOGON",
        "/RL", "LIMITED",
        "/F"
    )
    & schtasks.exe @SchTasksArgs
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks.exe failed with exit code $LASTEXITCODE. Run this script from an elevated PowerShell window."
    }
    $Registered = $true
}

Write-Host "Registered task: $TaskName"
Write-Host "Python: $PythonExe"
Write-Host "WorkingDirectory: $RepoRoot"
Write-Host "Arguments: $Arguments"
Write-Host "Start manually with: Start-ScheduledTask -TaskName `"$TaskName`""
