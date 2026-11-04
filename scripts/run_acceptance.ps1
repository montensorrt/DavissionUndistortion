# ===================================================================
# C++ 全量验收（原生层口径复现）：demo/raw -> demo/output-trt，与 CPU 版本逐像素比对
#   - 40 张图（可 -Limit 截断）
#   - 输出 acceptance_report.json / acceptance_per_image.csv
#   - 覆盖 pinned / pageable / 零拷贝 / 纯 kernel 四种口径
#
# 说明：这是原生能力层的验收程序（trt\tests\test_undist_api.cpp），
#       C# 交付层对应的精简验收用 scripts\run_verify.ps1。
# ===================================================================
param(
    [int]$Limit = 0,
    [int]$Iters = 5,
    [int]$PageableImages = 8,
    [int]$KernelIters = 100,
    [int]$Log = 1
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $root
$v0 = Join-Path $workspace "undistortion_standaloneV0"
$exe = Join-Path $root "trt\build\test_undist_api.exe"
if (-not (Test-Path $exe)) { throw "未找到 $exe，请先运行 scripts\build_native.ps1" }

. (Join-Path $PSScriptRoot "_env.ps1")

# 只有 C++ 验收程序的 BMP I/O 需要 OpenCV 运行时
$opencvBin = "E:\3rdparty\opencv\build\x64\vc16\bin"
if (Test-Path $opencvBin) { $env:PATH = "$opencvBin;$env:PATH" }

$matrix = Join-Path $v0 "botong25"
$raw = Join-Path $v0 "demo\raw"
$cpu = Join-Path $v0 "demo\output0"
$out = Join-Path $v0 "demo\output-trt"
$json = Join-Path $root "document\data\acceptance_report.json"
$cache = Join-Path $root "trt\build\cache\undist_5120x4096.engine"

& $exe --matrix $matrix --raw $raw --cpu $cpu --out $out --json $json `
    --engine-cache $cache --limit $Limit --iters $Iters `
    --pageable-images $PageableImages --kernel-iters $KernelIters --log $Log
exit $LASTEXITCODE
