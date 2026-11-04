# DavissionUndistortion —— 去畸变（CPU 版本 / StandaloneV0）

> **本分支 `cpu-v0` = CPU 版本**：纯 CPU 实现（**AVX2 + FMA** SIMD），5120×4096 单帧
> **63.82 ms**（15.5 张/s），与参考实现 **bit-exact（0 差异）**。
>
> **GPU 版本**（TensorRT 10.14 + CUDA 12.8，单帧 4.94 ms / **12.9×** CPU）在本仓库的
> **`main` 分支** → <https://github.com/montensorrt/DavissionUndistortion/blob/main/README.md>
>
> 本分支是 `Algorithms\undistortion_standaloneV0` 的**白名单快照，单向同步**：
> 真值源在工程目录，改本仓库不会回流。规则见 [`UPLOAD_RULES.md`](UPLOAD_RULES.md)。

| 项 | 值 |
|---|---|
| 分支 | `cpu-v0`（CPU 版本 / StandaloneV0） |
| 标签 | `26452`（附注标签） |
| 业务文件 | **65** 个 / 约 **0.67 MB** |
| 技术栈 | C++17（`/arch:AVX2` + `__FMA__`）· C ABI · C#（**net6.0**，OpenCvSharp4 4.8）· Python 工具 |
| 性能 | 5120×4096：infer 均值 **63.82 ms**（中位 64.63 ms），**15.5 张/s** |
| 验收 | 40 张实测图 / 838.86 MPix 逐像素比对：**bit-exact，0 差异** |
| 数据 | **不随仓库提供**：`botong25`（12 × 80 MiB 矩阵）、`demo/`（2.4 GB 图）需自备 |

## 这是什么

把「预计算 4-tap 双线性 LUT 重映射」的去畸变算法做成一个可独立运行、可被三种方式调用的库：

```
out = floor( clamp( Σ src[idx_k] * w_k , 0, 255 ) + 0.5 )
```

- 权重矩阵：`U_1..U_4 / V_1..V_4 / A1..A4` 共 **12 × 80 MiB**（5120×4096 float32，列主序）；
- 边界语义：四抽头「**全有或全无**」——任一越界则该像素取哑元 0；
- 数值语义：`clamp → +0.5 → trunc`，FMA 链与参考实现（`reference/Undistortion_raw.cpp`）一致；
- 三层封装：**C++ 核心** → **C ABI**（`Undistortion_C_API.h`）→ **C# 封装**（P/Invoke）。

## 目录与各子工程的角色

```
CAPP/                  ← C++ 核心（CMake 工程）★ 真源码
├── CMakeLists.txt / CMakeSettings.json / README.md
├── include/           CommonTypes.h · Logger.h · Undistortion.h（+ 三个 C_API 头）
├── src/               Undistortion.cpp（核心）· Logger.cpp · C_API 桥接
├── test/              test_undistortion.cpp · bench_undistortion.cpp
├── document/          9 份性能/优化分析（AVX2、FMA、LUT、回归分析…）
├── prompt/            default.md（需求原文）
└── scripts/           compare_images.py（逐像素比对）
CSharpApp/             ← C# 封装与示例（net6.0）
├── Undistor.sln
└── Undistor/          NativeAPI 三件套 + UndistortionProcessor/Runner
document/              ← UNDIST-001（现状与性能总结）· UNDIST-002（TRT 化方案预估）
tools/                 ← 20 个 Python 探针/基准脚本（矩阵识别、耗时拆解、TRT 试算…）
reference/             ← Undistortion_raw.cpp（复刻参考实现，bit-exact 基准）
prompt/                ← 3 份需求留档
```

## 快速开始

```powershell
# 1) C++ 构建（CMake + VS2022，需 AVX2）
cd CAPP
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# 2) 跑测试（矩阵与图片需自备，见“数据”）
build\Release\test_undistortion.exe --matrix ..\botong25 --image ..\demo\raw\PIC_1_1.bmp

# 3) C# 封装：先完成第 1 步（csproj 引用 CAPP\build\Undistortion.vcxproj），
#    再补 CSharpApp\ReferenceDLL\opencv_world4120.dll，然后用 VS 打开 Undistor.sln
```

**运行时依赖**

| 依赖 | 说明 |
|---|---|
| MSVC 2015-2022 x64 运行时 | C++ 库（`MSVCP140` / `VCRUNTIME140(_1)`） |
| AVX2 + FMA | 2013 年后的 Intel/AMD CPU 均支持；不支持会崩在非法指令 |
| OpenCV 4.12（C# 侧） | `opencv_world4120.dll`（约 61 MB，**本仓库不提供**，放 `CSharpApp\ReferenceDLL\`） |
| .NET 6 运行时 + OpenCvSharp4 4.8 | C# 示例程序（NuGet 还原） |

> ⚠️ 本仓库不含 `CAPP/build/`（CMake 产物，17 MB）。克隆后**必须先执行第 1 步**生成
> `Undistortion.vcxproj`，否则 `Undistor.sln` 无法打开/编译（它的 ProjectReference 指向该文件）。

## 快照统计（本次同步实测）

| 项 | 值 |
|---|---|
| 仓库文件总数 | **69**（业务 65 + 根自述 `README.md`、`UPLOAD_RULES.md`、`.gitignore`、`.gitattributes`） |
| 体积 | 约 **0.68 MB**（最大单文件 `CAPP/src/Undistortion.cpp` 57 KB） |
| C++ | `CAPP/` 28 文件（6 头 + 5 源 + 2 测试 + 9 文档 + 工程文件 + 脚本） |
| C# | `CSharpApp/` 10 文件（1 sln + 7 源/工程 + 2 需求文档） |
| Python | `tools/` 20 个脚本 |
| 文档 | 2 份 HTML（UNDIST-001/002）+ 9 份 Markdown（CAPP/document） |
| 同步时间 | 2026-11-04 |

**已排除**：`botong25/`（960 MB 权重矩阵）、`demo/`（2.4 GB 测试图）、
`document/data/`（256 MB 实测数据）、`CSharpApp/ReferenceDLL/`（182.6 MB 第三方 DLL：
`opencv_world4120.dll` / `opencv_world4120d.dll`）、`CSharpApp/.vs/`（162.2 MB）、
`build/`、`CAPP/build/`（17.1 MB）、`.git/`、`.vscode/`、`tools/__pycache__/`、
`Undistor.csproj.lscache`。逐项规则见 [`UPLOAD_RULES.md`](UPLOAD_RULES.md)。

## 与 GPU 版本的关系

同仓库 `main` 分支是这条 CPU 线的 **TensorRT 化** 结果（`Algorithms\undistortion`）：

| 口径 | CPU（本分支） | GPU（`main`） | 倍数 |
|---|---:|---:|---:|
| 单帧（5120×4096） | 63.82 ms | 4.94 ms | **12.9×** |
| 吞吐 | 15.5 张/s | 202 张/s | — |
| 一致性 | 基准（bit-exact） | 与 CPU 逐像素 ±1 LSB、差异 5.858e-6 | — |

迁移动因、方案对比与预估见 `document/UNDIST-002`；GPU 侧最终实现与验收见 `main` 分支。
