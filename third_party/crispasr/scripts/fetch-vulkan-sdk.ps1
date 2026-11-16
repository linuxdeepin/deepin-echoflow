# Download the Windows SDK without exposing a partial installer to callers.
# Invoke-WebRequest's built-in retries only cover HTTP status failures;
# the release failure was a connection reset while reading the installer.
[CmdletBinding()]
param(
    [string]$VersionUri = 'https://vulkan.lunarg.com/sdk/latest.json',
    [string]$DownloadRoot = 'https://sdk.lunarg.com/sdk/download',
    [string]$OutFile = 'vulkan_sdk.exe',
    [ValidateRange(1, 10)][int]$MaxAttempts = 5,
    [ValidateRange(0, 60)][int]$RetryDelaySeconds = 3
)

$ErrorActionPreference = 'Stop'

function Invoke-SdkRequest {
    param([scriptblock]$Request, [string]$Description)
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            & $Request
            return
        }
        catch {
            if ($attempt -eq $MaxAttempts) {
                throw
            }
            Write-Warning "$Description failed (attempt $attempt/$MaxAttempts): $($_.Exception.Message)"
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
}

$metadata = Invoke-SdkRequest {
    Invoke-RestMethod -Uri $VersionUri -TimeoutSec 60
} 'Vulkan SDK version lookup'
$version = $metadata.windows
if ($version -notmatch '^\d+\.\d+\.\d+\.\d+$') {
    throw "Invalid Windows Vulkan SDK version in $VersionUri"
}

$destination = [System.IO.Path]::GetFullPath($OutFile)
$partial = "$destination.partial"
$url = "$($DownloadRoot.TrimEnd('/'))/$version/windows/vulkan_sdk.exe"
Invoke-SdkRequest {
    try {
        Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 600
        if ((Get-Item -LiteralPath $partial).Length -eq 0) {
            throw 'Vulkan SDK download is empty'
        }
        Move-Item -LiteralPath $partial -Destination $destination -Force
    }
    finally {
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force
        }
    }
} 'Vulkan SDK installer download'

# The workflows capture this version to locate C:\VulkanSDK\<version>.
Write-Output $version
