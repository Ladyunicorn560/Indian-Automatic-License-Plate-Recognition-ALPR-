# Get the directory of the current script
$ProjectDir = $PSScriptRoot
if (-not $ProjectDir) { $ProjectDir = Get-Location }

$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath("Desktop")
$ShortcutPath = "$DesktopPath\Indian ALPR.lnk"

# Only create if it doesn't already exist or if we want to refresh it
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "$ProjectDir\run_app.bat"
$Shortcut.WorkingDirectory = "$ProjectDir"
$Shortcut.IconLocation = "powershell.exe"
$Shortcut.Description = "Launch Indian ALPR Streamlit App"
$Shortcut.Save()

Write-Host "Shortcut 'Indian ALPR' updated/created on your desktop pointing to: $ProjectDir" -ForegroundColor Green
