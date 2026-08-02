param(
    [int]$NTrials = 50
)

$workDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $workDir ".omo/logs/hpo-run.log"
$venvPython = Join-Path $workDir ".venv/Scripts/python.exe"
$hpoScript = Join-Path $workDir "scripts/hpo.py"

# Ensure log directory
$null = New-Item -ItemType Directory -Path (Split-Path $logFile -Parent) -Force

# Start timestamp
"=== HPO RUN STARTED: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $logFile -Encoding utf8
"=== n_trials = $NTrials ===" | Out-File $logFile -Encoding utf8 -Append

# Run and capture output progressively
Set-Location $workDir
$env:PYTHONIOENCODING = "utf-8"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $venvPython
$psi.Arguments = "scripts/hpo.py experiment=yolo26m --n-trials $NTrials"
$psi.WorkingDirectory = $workDir
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
$psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"
$psi.EnvironmentVariables["PATH"] = [Environment]::GetEnvironmentVariable("PATH")

$proc = [System.Diagnostics.Process]::Start($psi)

# Read stdout asynchronously
$stdoutReader = [System.Threading.Tasks.Task]::Run({
    $reader = $proc.StandardOutput
    while (($line = $reader.ReadLine()) -ne $null) {
        $line | Out-File $logFile -Encoding utf8 -Append
        Write-Host $line
    }
})

# Read stderr asynchronously
$stderrReader = [System.Threading.Tasks.Task]::Run({
    $reader = $proc.StandardError
    while (($line = $reader.ReadLine()) -ne $null) {
        "[STDERR] $line" | Out-File $logFile -Encoding utf8 -Append
        Write-Host "[STDERR] $line"
    }
})

# Wait for completion
$proc.WaitForExit()
$stdoutReader.Wait()
$stderrReader.Wait()

# End timestamp
"=== HPO RUN COMPLETED: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $logFile -Encoding utf8 -Append
"=== EXIT CODE: $($proc.ExitCode) ===" | Out-File $logFile -Encoding utf8 -Append

exit $proc.ExitCode
