# Local refresh with bounded retries, exponential backoff, and an overall timeout.
# season and data-through are NOT passed here -- publish.py derives both from
# STUFFPLUS_YEARS (season = later year) and the actual max game date in the
# data for that season year. Set STUFFPLUS_YEARS in webapp_publisher\.env to
# control which season is graded.
param(
  [int]$MaxRetries = 4,
  [int]$TimeoutMinutes = 30,
  [string]$GameTree = $env:STUFFPLUS_GAME_TREE,
  [string]$SummaryPath = $env:STUFFPLUS_SUMMARY,
  [string]$Years = $env:STUFFPLUS_YEARS,
  [switch]$SkipPipeline
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
    if (-not $SkipPipeline) {
      if (-not $GameTree -or -not $SummaryPath -or -not $Years) {
        throw "GameTree, SummaryPath and Years are required unless -SkipPipeline is passed"
      }
      if (-not $env:STUFFPLUS_WORKDIR) { throw "STUFFPLUS_WORKDIR must be set" }
      # Deterministic filename so the scorer stage can predict the path. The
      # pipeline's own default is wall-clock based and unusable from a schedule.
      $targetCsv = Join-Path $env:STUFFPLUS_WORKDIR "Final_Target_Calc_current.csv"
      $pipeOut = Join-Path $logDir "refresh-$dateStamp-attempt$attempt-pipeline.log"
      $pipeErr = Join-Path $logDir "refresh-$dateStamp-attempt$attempt-pipeline.err.log"
      $pipeProc = Start-Process -FilePath "python" -ArgumentList @(
        "python_files\target_and_calculated_pipeline.py",
        "--base-path", $GameTree,
        "--years", $Years,
        "--summary-path", $SummaryPath,
        "--out-dir", $env:STUFFPLUS_WORKDIR,
        "--out-name", "Final_Target_Calc_current.csv"
      ) -NoNewWindow -PassThru -RedirectStandardOutput $pipeOut -RedirectStandardError $pipeErr
      if (-not $pipeProc.WaitForExit($remainingSec * 1000)) {
        try { $pipeProc.Kill() } catch {}
        throw "target pipeline timed out after ${remainingSec}s (attempt $attempt)"
      }
      if ($pipeProc.ExitCode -ne 0) {
        throw "target pipeline exited with code $($pipeProc.ExitCode) (attempt $attempt)"
      }
      # publish.py reads STUFFPLUS_DATA; point it at what we just built.
      $env:STUFFPLUS_DATA = $targetCsv
      # Recompute the remaining budget so publish gets the time actually left.
      $remainingSec = [int]([Math]::Floor(($deadline - (Get-Date)).TotalSeconds))
      if ($remainingSec -le 0) { throw "no time left for publish after pipeline (attempt $attempt)" }
    }
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
