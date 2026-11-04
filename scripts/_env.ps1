# ===================================================================
# 运行期环境：把 TensorRT / CUDA 运行库目录加入 PATH。
# 这些 DLL 体积很大（nvinfer_10.dll 约 412 MB），不随仓库上传；
# 各 run_*.ps1 用 `. $PSScriptRoot\_env.ps1` 引入。
# ===================================================================

$trtBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\v10.14.1.48\bin"
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

foreach ($dir in @($trtBin, $cudaBin)) {
    if (-not (Test-Path $dir)) {
        Write-Warning "运行库目录不存在: $dir（请按实际安装路径修改 scripts\_env.ps1）"
    }
}

$env:PATH = "$trtBin;$cudaBin;$env:PATH"
