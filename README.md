# DavissionUndistortion —— 去畸变（GPU 版本）

> **本仓库是 `Algorithms\undistortion` 的白名单快照，单向同步**：真值源在工程目录，
> 改本仓库不会回流。目录规则见 [`UPLOAD_RULES.md`](UPLOAD_RULES.md)。
>
> **当前分支 `main` = GPU 版本**：用 **NVIDIA TensorRT 10.14 + CUDA 12.8** 在 GPU 上做去畸变，
> 单帧 ≈ **4.94 ms**（vs CPU 63.82 ms，**12.9×**）。CPU 版本（`undistortion_standaloneV0`，AVX2+FMA，**63.82 ms**）在同一仓库的 **`cpu-v0` 分支**
> （标签 `26452`），见 <https://github.com/montensorrt/DavissionUndistortion/tree/cpu-v0>。

| 项 | 值 |
|---|---|
| 分支 | `main`（GPU 版本）· `cpu-v0`（CPU 版本） |
| 标签 | `26453`（附注标签） |
| 业务文件 | **50** 个（仓库文件共 **52** 个）/ 约 **1.9 MB** |
| 交付物 | `undist_api.dll`（C ABI，88 KB）+ `Undistortion.Inference.dll`（.NET 8 类库）+ Test/Tools 两个可执行程序 |
| 运行时依赖 | 驱动 **581.57** / CUDA **12.8** / TensorRT **10.14.1** / VC++ 2015-2022 x64（**不需要 cuDNN**） |
| 验收 | 40 图逐像素 **PASS**：最大 ±1 LSB、差异 5.858×10⁻⁶、无 ≥2 差异、PSNR ≥ 99.47 dB |

## 先读哪一篇

| 场景 | 文档 |
|---|---|
| 新机器从零装环境（驱动 / CUDA / TRT / cuDNN） | [`csharp/document/UNDIST-108`](csharp/document/UNDIST-108-去畸变Undistortion_第三方库安装指南_显卡驱动_CUDA_TensorRT_cuDNN_20261104.html) |
| 5 分钟把一张图跑出去 | [`csharp/document/UNDIST-107`](csharp/document/UNDIST-107-去畸变Undistortion_最小开始_Test调用说明_20261104.html) |
| 上位机接入 / 部署形态 / 排障 | [`csharp/document/UNDIST-106`](csharp/document/UNDIST-106-去畸变Undistortion_项目使用说明_目录结构_环境配置与集成_20261104.html) |
| 项目结构为什么长这样、为什么只留 Mat | [`document/UNDIST-105`](document/UNDIST-105-去畸变Undistortion_项目梳理_结构与功能保留去除分析_20261104.html) |
| TRT 推理原理 / 验收数据 / 内存搬运 | [`document/UNDIST-101~103`](document/) |

## 快照统计（本次同步实测）

| 项 | 值 |
|---|---|
| 仓库文件总数 | **52**（业务 50 + 根自述 `UPLOAD_RULES.md`、`.gitattributes`） |
| 体积 | 约 **1.9 MB**（`git ls-files` 量级；最大单文件 755 KB） |
| C# | 3 个项目 / 18 文件（5 类库源 + 2 Test + 2 Tools + `thirdpart/undist_api.dll` + 3 文档 + props + sln） |
| 原生 | `trt/` 8 文件（CMake + 3 头 + 3 源 + 1 验收程序） |
| 文档 | 8 份 HTML（101~108）+ 2 张效果图 + 3 个实测数据文件 |
| 脚本 | 7 个（6 ps1 + 1 py） |
| 需求留档 | 4 个 md |
| 最大单文件 | `document/assets/undis_ee2_before_after.jpg` = 755,137 B |
| 同步时间 | 2026-11-04 |

**已排除**：`trt/build/`（29 文件 / 8 MB，构建产物 + 引擎缓存）、
`document/data/preview/`（20 MB 预览图）、`prompt/github.md`（上传指令，含凭据路径）、
`csharp/**/bin|obj|output`、`.vs/`。逐项规则见 [`UPLOAD_RULES.md`](UPLOAD_RULES.md)。

---

## 工程 README（原文）
### Undistortion —— 去畸变 TensorRT 推理
把 `Algorithms/undistortion_standaloneV0` 的 CPU 去畸变算法迁移为 **TensorRT 推理**，
封装为 **一个 DLL（C ABI）+ 一个 .NET 8 类库 + 两个可直接运行的 .NET 程序**：

- **初始化**：参数是值类型结构体 `UndistortionConfig`（矩阵目录 / 宽高 / 设备号），
  上位机在代码里直接构造 —— **没有配置文件，没有第二个地方要配**；
- **推理**：`Mat` 进 → `Mat` 出，全程内存态，不落盘；
- **日志固定关闭**：日志只服务离线调用，上位机不启用；
- **引擎缓存固定关闭**：交付层每次现场构建（≈6.7 s，进程启动一次），缓存只在离线 C ABI 侧用；
- **数值**：与 CPU Release 版本逐像素一致（40 图验收：最大 ±1 LSB、差异 5.858×10⁻⁶、无 ≥2 差异、PSNR ≥ 99.47 dB）。

| 文档 | 内容 |
|---|---|
| [UNDIST-101](document/UNDIST-101-去畸变UndistortionV1_TRT推理与CSharp封装_设计与实现_20261026.html) | TRT 推理与 C# 封装 设计与实现 |
| [UNDIST-102](document/UNDIST-102-去畸变UndistortionV1_TRT推理_测试与验收报告_20261026.html) | TRT 推理 测试与验收报告（40 图） |
| [UNDIST-103](document/UNDIST-103-去畸变UndistortionV1_内存搬运与批量耗时_测试分析与总结_20261026.html) | 内存搬运口径与批量耗时 |
| [UNDIST-104](document/UNDIST-104-去畸变UndistortionV1_CSharp接口文档_快速开始与详细参考_20261026.html) | C# 接口文档（**完整能力面**，含已收敛的入口） |
| [UNDIST-105](document/UNDIST-105-去畸变Undistortion_项目梳理_结构与功能保留去除分析_20261104.html) | **项目梳理与精简重构**：结构、保留/去除、结构体配置、日志/缓存策略、命名、去除能力档案 |
| [UNDIST-106](csharp/document/UNDIST-106-去畸变Undistortion_项目使用说明_目录结构_环境配置与集成_20261104.html) | **项目使用说明**：目录结构、依赖三分类、环境配置、上位机接入、FAQ、文档索引 |
| [UNDIST-107](csharp/document/UNDIST-107-去畸变Undistortion_最小开始_Test调用说明_20261104.html) | **最小开始**：只讲 Undistortion.Test 的配置参数 / 初始化 / 调用 / 释放（85 行） |
| [UNDIST-108](csharp/document/UNDIST-108-去畸变Undistortion_第三方库安装指南_显卡驱动_CUDA_TensorRT_cuDNN_20261104.html) | **第三方库安装指南**：显卡驱动 / CUDA / TensorRT / cuDNN 安装与环境变量 |

> 交付层（C# 类库）只保留 **Mat 一种入口**，只认 **一个结构体**；`byte[]`、零拷贝 `PinnedBuffer`、
> 裸指针、纯 kernel 基准、内存模式切换、日志开关、引擎缓存开关都已在交付层移除，
> 其实现要点与实测数据完整保留在 UNDIST-104 / UNDIST-105 中。

## 目录

```
undistortion/
├── csharp/                       # .NET 8（三个项目，一个解决方案）
│   ├── Undistortion.sln
│   ├── Directory.Build.props     # 公共属性 + 原生 DLL 自动定位/拷贝
│   ├── Undistortion.Inference/   # 主项目：类库（Mat 进 → Mat 出）
│   │   ├── UndistortionConfig.cs       # ★ 初始化参数结构体（唯一配置载体）
│   │   ├── Undistorter.cs              # 句柄生命周期 + Mat 推理
│   │   ├── UndistortionInfo.cs         # Info + Timing
│   │   ├── UndistortionException.cs
│   │   └── NativeMethods.cs            # 5 个 P/Invoke
│   ├── Undistortion.Test/        # 测试项目：加载指定图像 → 初始化 → 执行 → 释放
│   ├── Undistortion.Tools/       # 工具项目：批量验收 + 与 CPU 参考图逐像素比对
│   ├── thirdpart/                # ★ 第三类依赖：我们自己生成的 undist_api.dll（入库）
│   └── document/                 # ★ C# 侧文档（UNDIST-106 使用说明）
├── trt/                          # 原生库（CMake + Ninja）
│   ├── include/undist_api.h      # C ABI（10 导出）
│   ├── src/undist_tables.cpp     # 12 个 .bin → idx0/fu/fv（12 B/px）
│   ├── src/undist_engine.cpp     # TRT 引擎 + CUDA 内存 + 三种搬运路径
│   ├── src/undist_api.cpp        # 句柄/锁/错误串/页锁注册
│   └── tests/test_undist_api.cpp # 原生层全量验收程序（历史口径复现）
├── scripts/                      # 构建 / 最小测试 / 验收 / 批处理
├── document/                     # 设计文档、测试报告、数据（UNDIST-101~105）
└── prompt/                       # 需求归档
```

## 依赖三分类

| 类别 | 依赖 | 获取方式 |
|---|---|---|
| ① 系统安装 | TensorRT 10.14.1.48、CUDA 12.8、VC++ 2015-2022 x64 | 官方安装包 + 配 `PATH` |
| ② NuGet | OpenCvSharp4 / OpenCvSharp4.runtime.win 4.13.0.20260627 | `PackageReference` 自动还原 |
| ③ thirdpart | `undist_api.dll`（本仓库构建，88 KB） | 已在 `csharp/thirdpart/`，拷走即用 |

详见 [UNDIST-106](csharp/document/UNDIST-106-去畸变Undistortion_项目使用说明_目录结构_环境配置与集成_20261104.html)。

## 依赖

| 依赖 | 版本 | 说明 |
|---|---|---|
| Visual Studio 2022 (MSVC) | 19.44 | `/O2 /MD /utf-8` |
| CUDA | 12.8 | `cudart64_12.dll` |
| TensorRT | 10.14.1.48 | `nvinfer_10.dll` |
| OpenCV | 4.12 | 仅原生层验收程序的 BMP I/O |
| .NET | 8.0 | C# 封装；OpenCvSharp4 4.13 |

TensorRT / CUDA 的 DLL 体积很大（`nvinfer_10.dll` 约 412 MB），不随仓库上传；运行前
由 `scripts/_env.ps1` 把安装目录加入 `PATH`。

## 构建与运行

```powershell
cd Algorithms\undistortion

# 1) 构建原生库 + C++ 验收程序
powershell -ExecutionPolicy Bypass -File scripts\build_native.ps1

# 2) 构建 C# 解决方案并跑最小调用（默认 demo\raw\PIC_1_1.bmp + botong25）
powershell -ExecutionPolicy Bypass -File scripts\run_test.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_test.ps1 -Image D:\a.bmp -Matrix D:\botong25

# 3) 验收比对（40 图，与 CPU 产物逐像素比对，写 document\data\csharp_verify.json）
powershell -ExecutionPolicy Bypass -File scripts\run_verify.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_verify.ps1 -Limit 3

# 4) 原生层全量验收（pinned / pageable / 零拷贝 / 纯 kernel 四种口径）
powershell -ExecutionPolicy Bypass -File scripts\run_acceptance.ps1

# 5) 数据集批处理（Python，按子目录保持同名同结构）
python scripts\undist_batch.py --raw <raw 目录> --out <输出目录>
```

**离线参数只来自命令行/脚本**：`run_test.ps1` / `run_verify.ps1` 各带一个 `-Matrix`（默认
`undistortion_standaloneV0\botong25`），尺寸由图像本身决定 —— 代码里没有配置文件，
也没有写死的 5120×4096。

**矩阵目录的默认值**：`Undistortion.Test` / `Undistortion.Tools` 的 `--matrix` 可省略，省略时用
`<exe 所在目录>\botong25`；给了相对名（如 `botong25`）也**按 exe 目录解析**（不看当前工作目录，
避免快捷方式/服务/计划任务启动时找不到）。开发期不想拷 960 MB 就用
`powershell -File scripts\link_matrix.ps1` 在输出目录建一个目录联接。
部署时把 `botong25` 拷到 exe 同级即可，上位机无需任何参数。

## 初始化参数：C# 结构体（唯一配置）

`csharp/Undistortion.Inference/UndistortionConfig.cs`，值类型、只读、`init` 赋值：

```csharp
var config = new UndistortionConfig
{
    MatrixDir = @"D:\opt\undistortion\botong25",   // 12 个 .bin 所在目录
    Width     = 5120,
    Height    = 4096,
    DeviceId  = 0,
};
using var undistorter = new Undistorter(config);
```

| 字段 | 对应原生 | 说明 |
|---|---|---|
| `MatrixDir` | `undist_create(matrix_dir_utf8)` | 含 U_1..4 / V_1..4 / A1..4 |
| `Width` / `Height` | `undist_create(width, height)` | 与矩阵尺寸一致（离线示例取 `Mat.Width/Height`） |
| `DeviceId` | `opt.device_id` | CUDA 设备号 |

路径原样交给原生库（相对路径按当前工作目录解析）——上位机请给**绝对路径**。

**不含日志字段、不含缓存字段**：两者都是“交付层固定关闭、离线侧按需开启”，见下两节。

## 日志：上位机不启用

- 交付层固定 `log_level = 0`，结构体里**没有**日志开关，避免误开；
- 0 档仍会打印**必须可见的告警**（页锁内存不足回退 pageable、缓存写失败、缓存头不匹配回退构建）；
- 1 = 关键节点、2 = 逐阶段明细，只有离线调用方开：C++ 验收 `--log 1|2`、Python 批处理
  `UndistOptions(log_level=1)`，或直接用 C ABI 的 `undist_options.log_level`；
- 上位机关闭的理由：日志写 stderr（GUI/服务无控制台，重定向成本不可控）、落在初始化
  （1.6 s ~ 7.5 s）的关键路径上、会绕过上位机自己的日志体系；
- 排障改用结构化数据：`Undistorter.Info`、`Undistorter.LastTiming`、`UndistortionException.Code` + 原生错误串。

## 引擎缓存：交付层不启用

实测（同一进程模板、同一张图，只改有没有缓存）：**不缓存 8.51 s / 命中缓存 1.79 s，差 ≈6.7 s**。

- 上位机「启动一次、长期运行」，这 6.7 s 是一次性成本，所以**交付层固定不缓存**：
  结构体没有该字段，`undist_create(..., engine_cache_path) = null`，每次现场构建；
- 好处是彻底没有路径 / 权限 / 并发 / 版本不匹配四类部署问题，也没有“这次为什么快或慢”的黑盒；
- 缓存能力仍在 **C ABI**：`undist_create` 第 5 个参数 + C++ 验收的 `--engine-cache`、
  Python 批处理的 `cache` 目录。离线工具反复跑，缓存能省 6.7 s/次，值得用；
- 缓存文件 37 KB、不含图像，32 B 头校验（尺寸 / 表变体 / TRT 版本 / 表字节数），不匹配自动重建；
- 表本身（240 MiB）**从来不缓存**，每次从 12 个 `.bin` 重建（1.3~2.1 s），这是初始化的下限。

## C# 速览

```csharp
using OpenCvSharp;
using Undistortion.Inference;

using Mat src = Cv2.ImRead("raw.bmp", ImreadModes.Grayscale);

var config = new UndistortionConfig                 // 1) 初始化参数（结构体）
{
    MatrixDir = @"D:\opt\undistortion\botong25",
    Width = src.Width,                              // 2) 尺寸来自图像，不写死
    Height = src.Height,
};

using var undistorter = new Undistorter(config);    // 3) 初始化（表 240 MB + 显存 280 MB）
using Mat dst = undistorter.Process(src);           // 4) 执行（~5 ms/图）
Cv2.ImWrite("undis.bmp", dst);                      // 5) using 退出即释放
```

高频路径复用输出缓冲，避免每帧 20 MB 分配：

```csharp
using var dst = new Mat(config.Height, config.Width, MatType.CV_8UC1);
undistorter.Process(src, dst);                      // 写入复用缓冲
Console.WriteLine(undistorter.LastTiming);          // h2d / infer / d2h / total
```

## C ABI 速览

```c
undist_options opt{1, 0, 1, 0};              // memory_mode, device, channels, log（log 仅离线用）
undist_ctx* ctx = nullptr;
undist_create("…/botong25", 5120, 4096, &opt, "…/cache/undist_5120x4096.engine", &ctx);

undist_timing t{};
undist_process(ctx, src, 5120, dst, 5120, 0, &t);
undist_release(ctx);
```

## 关键结论（RTX 5060 Laptop / TRT 10.14 / 5120×4096）

| 口径 | 每帧 | 图/s | vs CPU 63.82 ms |
|---|---:|---:|---:|
| GPU 直采（图已在显存） | 1.198 ms | 835 | 53.3× |
| 调用方 pinned 零拷贝 | 2.892 ms | 346 | 22.1× |
| **库内 pinned 暂存（交付层 Mat 路径）** | **4.944 ms** | **202** | **12.9×** |
| pageable 直传 | 5.904 ms | 169 | 10.8× |

- 表：变体 F（`idx0` int32 + `fu`/`fv` fp32）= **12 B/px / 240 MiB**；显存 **280 MiB/实例**；
- 初始化：表 1.3~2.1 s + 引擎构建 ≈6.7 s（交付层不缓存）→ 进程启动约 8.5 s；
- 引擎绑定 GPU 架构 + TensorRT 版本，离线缓存 37 KB，可随时删除重建。

## 限制

- 仅单通道 `CV_8UC1`，尺寸在初始化时固定；
- 每实例一条 stream，不可并发（每线程一实例，显存 280 MB/实例）；
- 单图无 batch；批量需自定义 plugin（后续工作）。

