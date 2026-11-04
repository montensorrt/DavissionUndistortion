# thirdpart —— 我们自己生成的原生库

本目录**只**放「由本仓库构建产出、需要随交付一起走」的原生二进制。
第三方运行时（TensorRT / CUDA / OpenCvSharp）**不放这里**，按下表分三类获取。

## 依赖三分类

| 类别 | 依赖 | 获取方式 | 落点 |
|---|---|---|---|
| ① 系统安装 | TensorRT **10.14.1.48**、CUDA **12.8**、VC++ 2015-2022 x64 运行时 | 官方安装包装到系统，配 `PATH` | 系统路径 |
| ② NuGet | OpenCvSharp4 / OpenCvSharp4.runtime.win **4.13.0.20260627** | `PackageReference` 自动还原 | 构建输出 |
| ③ 本目录 | `undist_api.dll`（本仓库 CMake 构建） | `scripts/build_native.ps1` 自动同步 | `csharp/thirdpart/` |

一句话：**能装系统的装系统，能走 NuGet 的走 NuGet，只有“我们自己造的”才放这里。**

## undist_api.dll

| 项 | 值 |
|---|---|
| 来源 | `trt/`（CMake + Ninja + MSVC x64） |
| 大小 | 88,064 B |
| SHA256 | `9BF3B8CD3DCF7A9CDAFF8011FD479DBDE9A07436E5C2486B2E3F39FE996C11DE` |
| 直接依赖 | `nvinfer_10.dll`、`cudart64_12.dll`、`MSVCP140/VCRUNTIME140/VCRUNTIME140_1` |
| 更新方式 | `scripts/build_native.ps1` 构建成功后**自动**拷一份到这里 |

手工同步（一般不需要）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_native.ps1
# 或
Copy-Item trt\build\undist_api.dll csharp\thirdpart\ -Force
```

> 这个 DLL 是**构建产物但需要入库**：客户端不装 Visual Studio / CUDA SDK / CMake，拷走即可用。
> `.gitignore` 里已用 `!csharp/thirdpart/*.dll` 显式保住它。

## 不要放这里的东西

| 文件 | 体积 | 原因 |
|---|---:|---|
| `nvinfer_10.dll` | 412 MB | 超 GitHub 单文件 100 MB 硬限制；受 NVIDIA 许可约束 |
| `nvinfer_builder_resource_sm120_10.dll` | 360 MB | 同上；且与 GPU 架构绑定，只能由安装包提供 |
| `cudart64_12.dll` | ~0.5 MB | 随 CUDA 安装包，避免版本错配 |
| `OpenCvSharpExtern.dll` | 65 MB | 走 NuGet（`OpenCvSharp4.runtime.win`），不要手工拷贝 |
| `cudnn*_9.dll`（整套） | ≈1 GB | **本项目不需要 cuDNN**（网络无 Conv/RNN，实测隔离运行通过） |
| `cublas64_12 / cublasLt64_12` | 768 MB | 不需要；隔离测试未带，构建与推理照常 |
| `nvinfer_plugin_10 / nvinfer_dispatch_10 / nvinfer_lean_10` | 94 MB | 不需要（无 plugin / 非 lean 运行时） |

完整说明见 [`csharp/document/UNDIST-106-…项目使用说明…html`](../document/UNDIST-106-去畸变Undistortion_项目使用说明_目录结构_环境配置与集成_20261104.html)，
其中 3.2.1 给出了“不需要 cuDNN”的静态依赖证据与隔离实测结果。
