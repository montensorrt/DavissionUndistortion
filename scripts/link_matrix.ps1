# ===================================================================
# 在指定目录旁挂一个 botong25（目录联接 junction，不复制 960 MB）
#
#   powershell -ExecutionPolicy Bypass -File scripts\link_matrix.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\link_matrix.ps1 -Target D:\deploy\app
#
# 用途：让 Undistortion.Test.exe / Undistortion.Tools.exe 不带 --matrix 也能跑
#       （默认读 <exe 所在目录>\botong25）。
# 注意：只用于开发/联调；真实部署请把 botong25 拷到目标目录（安装包/部署脚本负责），
#       不要把 junction 当成交付物。
# ===================================================================
param(
    [string]$Target = "",
    [string]$Matrix = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $root
$v0 = Join-Path $workspace "undistortion_standaloneV0"

if ([string]::IsNullOrWhiteSpace($Matrix)) { $Matrix = Join-Path $v0 "botong25" }
if ([string]::IsNullOrWhiteSpace($Target)) {
    $Target = Join-Path $root "csharp\Undistortion.Test\bin\Release\net8.0"
}

if (-not (Test-Path $Matrix)) { throw "矩阵目录不存在: $Matrix" }
if (-not (Test-Path $Target)) { throw "目标目录不存在（先构建一次）: $Target" }

$link = Join-Path $Target "botong25"
if (Test-Path $link) {
    Write-Host "[skip] 已存在: $link" -ForegroundColor Yellow
    exit 0
}

New-Item -ItemType Junction -Path $link -Target $Matrix | Out-Null
Write-Host "[ok]   $link" -ForegroundColor Green
Write-Host "   ->  $Matrix" -ForegroundColor DarkGray
Write-Host "现在可以直接: Undistortion.Test.exe --image <图>   （不带 --matrix）" -ForegroundColor DarkGray
