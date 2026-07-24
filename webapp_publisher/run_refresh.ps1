# Local refresh with bounded retries, exponential backoff, and an overall timeout.
# season and data-through are NOT passed here -- publish.py derives both from
# STUFFPLUS_YEARS (season = later year) and the actual max game date in the
# data for that season year. Set STUFFPLUS_YEARS in webapp_publisher\.env to
# control which season is graded.
param(
  [int]$MaxRetries = 4,
  [int]$TimeoutMinutes = 30
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$dateStamp = (Get-Date).ToString("yyyy-MM-dd")
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$delay = 5
for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
  if ((Get-Date) -gt $deadline) { Write-Error "Refresh exceeded ${TimeoutMinutes}m timeout"; exit 1 }
  try {
    $remainingSec = [int]([Math]::Floor(($deadline - (Get-Date)).TotalSeconds))
    if ($remainingSec -le 0) { Write-Error "Refresh exceeded ${TimeoutMinutes}m timeout"; exit 1 }
    $stdoutLog = Join-Path $logDir "refresh-$dateStamp-attempt$attempt.log"
    $stderrLog = Join-Path $logDir "refresh-$dateStamp-attempt$attempt.err.log"
    $proc = Start-Process -FilePath "python" -ArgumentList @("-m","webapp_publisher.publish") -NoNewWindow -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    if (-not $proc.WaitForExit($remainingSec * 1000)) {
      try { $proc.Kill() } catch {}
      throw "publish timed out after ${remainingSec}s (attempt $attempt)"
    }
    if ($proc.ExitCode -ne 0) { throw "publish exited with code $($proc.ExitCode) (attempt $attempt)" }
    Write-Host "Refresh succeeded on attempt $attempt"; exit 0
  } catch {
    Write-Warning "Attempt $attempt failed: $_"
    if ($attempt -eq $MaxRetries) { Write-Error "Refresh failed after $MaxRetries attempts"; exit 1 }
    Start-Sleep -Seconds $delay; $delay = $delay * 2
  }
}
