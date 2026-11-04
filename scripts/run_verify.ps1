# ===================================================================
# 验收比对：批量去畸变 → 与 CPU Release 产物逐像素比对 → 写 JSON 报告
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_verify.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_verify.ps1 -Limit 3
#
# 离线参数（矩阵目录 / 数据目录）全部来自本脚本，代码里没有配置文件。
# ===================================================================
param(
    [int]$Limit = 0,
    [string]$Raw = "",
    [string]$Ref = "",
    [string]$Matrix = "",
    [string]$Json = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $root
$v0 = Join-Path $workspace "undistortion_standaloneV0"

if ([string]::IsNullOrWhiteSpace($Raw)) { $Raw = Join-Path $v0 "demo\raw" }
if ([string]::IsNullOrWhiteSpace($Ref)) { $Ref = Join-Path $v0 "demo\output0" }
if ([string]::IsNullOrWhiteSpace($Matrix)) { $Matrix = Join-Path $v0 "botong25" }
if ([string]::IsNullOrWhiteSpace($Json)) { $Json = Join-Path $root "document\data\csharp_verify.json" }

dotnet build (Join-Path $root "csharp\Undistortion.sln") -c Release --nologo | Out-Null
if ($LASTEXITCODE -ne 0) { throw "C# 构建失败" }

. (Join-Path $PSScriptRoot "_env.ps1")

$exe = Join-Path $root "csharp\Undistortion.Tools\bin\Release\net8.0\Undistortion.Tools.exe"
& $exe verify --raw $Raw --ref $Ref --matrix $Matrix --limit $Limit --json $Json
exit $LASTEXITCODE
