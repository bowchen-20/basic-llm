$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
Set-Location C:\Users\bowch\basic-llm

$status = git status --porcelain
if ($status) {
    $date = Get-Date -Format "yyyy-MM-dd"
    git add -A
    git commit -m "Daily progress update - $date"
    git push origin main
} else {
    Write-Output "No changes to commit."
}
