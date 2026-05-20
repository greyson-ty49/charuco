# Upload current folder to GitHub repo
param(
    [string]$RepoUrl = "https://github.com/greyson-ty49/charuco.git",
    [string]$Branch = "main",
    [string]$CommitMessage = "Initial commit"
)

$ErrorActionPreference = "Stop"

Write-Host "Repo: $RepoUrl"
Write-Host "Branch: $Branch"
Write-Host "Commit: $CommitMessage"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not installed or not on PATH. Install Git from https://git-scm.com/downloads and try again."
}

if (-not (Test-Path -Path . -PathType Container)) {
    Write-Error "Current directory is not a folder."
}

if (-not (Test-Path -Path ".git")) {
    git init | Out-Null
}

git add .

# If nothing to commit, git commit will fail; handle it gracefully
try {
    git commit -m $CommitMessage | Out-Null
} catch {
    Write-Host "Nothing to commit (working tree clean or only ignored files)."
}

# Ensure branch name
$CurrentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($CurrentBranch -ne $Branch) {
    git branch -M $Branch
}

# Set remote origin
$ExistingOrigin = ""
try {
    $ExistingOrigin = (git remote get-url origin 2>$null).Trim()
} catch {
    $ExistingOrigin = ""
}
if ([string]::IsNullOrWhiteSpace($ExistingOrigin)) {
    git remote add origin $RepoUrl
} elseif ($ExistingOrigin -ne $RepoUrl) {
    git remote set-url origin $RepoUrl
}

# Push
Write-Host "Pushing to $RepoUrl ..."
git push -u origin $Branch

Write-Host "Done."
