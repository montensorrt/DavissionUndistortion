# 规则说明

本文件是 `montensorrt/DavissionUndistortion` 仓库 **`cpu-v0` 分支（CPU 版本 / StandaloneV0）**
的上传边界与核验口径。基线沿用同仓库 `main` 分支的 `UPLOAD_RULES.md`
（源自 `Algorithms\masktoolV0\document\MaskTool_上传GitHub规则与逻辑_V2.html` 的白名单机制），
按本工程目录结构做最小必要替换。

| 项 | 值 |
|---|---|
| 真值源 | `<工作区>\Algorithms\undistortion_standaloneV0`（**它自己是一个 git 仓库**，但与本次上传无关，**不取它的 `.git`**） |
| 目标仓库 | `https://github.com/montensorrt/DavissionUndistortion.git` |
| 目标分支 | **`cpu-v0`**（新建；`main` 是 GPU 版本，两条线互不覆盖） |
| 目标标签 | `26452`（附注标签，用户指定） |
| 同步方向 | **单向**：工程目录 → 仓库。改代码改工程目录 |
| 仓库根 | 对应 `Algorithms\undistortion_standaloneV0\` 这一层 |

---

## 规则 1 —— 屏蔽路径（按目录名匹配，任何层级生效）

```text
.git/                 # 真值源自己的仓库，绝不嵌套上传
.vscode/              # 本机编辑器设置
build/                # 根级 CMake 产物（151 文件 / 0.8 MB）
CAPP/build/           # C++ 构建产物（17.1 MB，含 Undistortion.vcxproj / bin / obj）
**/bin/  **/obj/      # MSBuild 产物
CSharpApp/.vs/        # Visual Studio 用户态（162.2 MB）
**/__pycache__/       # Python 字节码
```

### 数据与二进制大件（按路径精确排除）

| 路径 | 文件数 | 体积 | 理由 |
|---|---:|---:|---|
| `botong25/` | 12 | **960 MB** | 去畸变权重矩阵（12 × 80 MiB），属数据资产，按需自备 |
| `demo/` | 123 | **2,460 MB** | 测试图（raw / output0 / output-trt / _repro_check） |
| `document/data/` | 36 | **255.9 MB** | 实测产物（大图 + 中间结果） |
| `CSharpApp/ReferenceDLL/` | 2 | **182.6 MB** | 第三方 DLL（`opencv_world4120.dll` / `d`），受 OpenCV 许可约束，按需自备 |
| `CSharpApp/.vs/` | — | 162.2 MB | VS 用户态 |
| `CAPP/build/` | — | 17.1 MB | CMake/Ninja/MSBuild 产物 |
| `build/` | 151 | 0.8 MB | 根级构建目录 |

---

## 规则 2 —— 工程根白名单

| 条目 | 取什么 | 理由 |
|---|---|---|
| `CAPP/` | 规则 3 列出的 28 个文件（**不含 `CAPP/build/`**） | C++ 核心 |
| `CSharpApp/` | `Undistor.sln` + `Undistor/`（7 源/工程）+ `prompt/`（2 份） | C# 封装 |
| `document/` | 仅 2 份 HTML（`UNDIST-001`、`UNDIST-002`） | 算法与性能结论 |
| `tools/` | 20 个 Python 脚本 / bat（不含 `__pycache__/`） | 基准与探针工具 |
| `reference/` | `README.md` + `Undistortion_raw.cpp` | 复刻参考实现（bit-exact 基准） |
| `prompt/` | 3 份需求留档 | 需求原文 |

### 规则 3 —— 逐类白名单

| 路径 | 文件 |
|---|---|
| `CAPP/`（根） | `CMakeLists.txt`、`CMakeSettings.json`、`README.md`、`.gitignore` |
| `CAPP/include/` | `CommonTypes.h`、`CommonTypes_C_API.h`、`Logger.h`、`Logger_C_API.h`、`Undistortion.h`、`Undistortion_C_API.h` |
| `CAPP/src/` | `Undistortion.cpp`、`Undistortion_C_API.cpp`、`Logger.cpp`、`Logger_C_API.cpp`、`CommonTypes_C_API.cpp` |
| `CAPP/test/` | `test_undistortion.cpp`、`bench_undistortion.cpp` |
| `CAPP/document/` | 9 份 Markdown（性能基准、AVX2/FMA/LUT 优化、回归分析…） |
| `CAPP/prompt/` | `default.md` |
| `CAPP/scripts/` | `compare_images.py` |
| `CSharpApp/`（根） | `Undistor.sln` |
| `CSharpApp/Undistor/` | `Undistor.csproj`、`Undistortion_NativeAPI.cs`、`CommonTypes_NativeAPI.cs`、`Logger_NativeAPI.cs`、`GenericNativeHandle.cs`、`UndistortionProcessor.cs`、`UndistortionRunner.cs` |
| `CSharpApp/prompt/` | `dingpai_prompt.md`、`feipai_measurement_awg.md` |
| `document/` | `UNDIST-001-…html`、`UNDIST-002-…html` |
| `prompt/` | `c_api.md`、`csharp_api.md`、`梳理分析.md` |
| `tools/` | 20 个：`build_release.bat`、`env_probe.py`、`exactness_audit.py`、`io_decompose.py`、`layout_check.py`、`make_preview.py`、`make_testset.py`、`map_probe.py`、`map_stats.py`、`mask_frame_check.py`、`matrix_identify.py`、`mem_to_mem_bench.py`、`reference_check.py`、`separable_check.py`、`trt_bench_final.py`、`trt_debug.py`、`trt_debug2.py`、`trt_undistort.py`、`undistort_batch.py`、`weight_recon.py` |
| `reference/` | `README.md`、`Undistortion_raw.cpp` |

---

## 规则 4 —— 其余一律不传

- `CAPP/build/`：虽然 C# 工程 `ProjectReference` 指向 `CAPP\build\Undistortion.vcxproj`，
  但那是 CMake 生成物 —— **不入库**，克隆后按 README 第 1 步重新生成；
- `CSharpApp/Undistor/Undistor.csproj.lscache`：语言服务缓存，可再生；
- 其余未列出的文件/目录，严格执行“不传”。

---

## 规则 5 —— 凭据红线（硬规则，无例外）

- 令牌位于**工作区级** `github\person-access-token.md`，不在本工程目录内，天然不遍历；
- 令牌**只允许**出现在推送命令的 `http.extraheader` 里；**不得**写入 `.git/config`、
  不得写进任何文件、不得进提交历史；
- 推送使用 `-c credential.helper=` 关闭凭据缓存；
- 推完自检：`git remote -v` 不含令牌；`git config -l | findstr -i token` 为空；
- 本仓库 `.gitignore` 含 `*access-token*`、`*secret*`、`*.pem`、`*.key` 作为第二道保险。

---

## 规则 6 —— 变更记录纪律

- 每次同步更新根 `README.md` 的“快照统计”块（文件数、体积、时间）；
- 标签只在内容冻结时打；本分支 `26452` 与 `main` 的 `26453` 各自独立，互不覆盖；
- 性能/优化结论写进 `CAPP/document/` 或 `document/`，不要只写在提交信息里。

---

## 上传逻辑（五步，与 `main` 分支一致）

```text
[1/5] 白名单枚举     65 个业务文件（不含 4 个根自述）
[2/5] 逐文件比对     新分支 → 全部为“新增”
[3/5] 敏感信息体检   凭据 0；无本机绝对路径写入源码
[4/5] 逐字节复制     复制到临时目录（工程目录不建 .git；也不带它自己的 .git）
[5/5] 写自述文件     README.md（CPU 版本说明）/ UPLOAD_RULES.md / .gitignore / .gitattributes
```

推送（令牌只在推送命令中出现，不落盘）：

```powershell
git init -b cpu-v0
git add -A
git commit -m "feat(cpu): Undistortion StandaloneV0 CPU 版本快照（AVX2+FMA，63.82ms/帧）"
git remote add origin https://github.com/montensorrt/DavissionUndistortion.git
git -c credential.helper= -c http.extraheader="AUTHORIZATION: basic <base64(x-access-token:TOKEN)>" push -u origin cpu-v0
git tag -a 26452 -m "Undistortion StandaloneV0 快照 · 26452（CPU 版本 / AVX2+FMA）"
git -c credential.helper= -c http.extraheader="AUTHORIZATION: basic <同上>" push origin 26452
```

---

## 核验清单

| # | 检查点 | 判据 | 本次实测 |
|---|---|---|---|
| 1 | 业务文件数 | 65 | ✅ 65 |
| 2 | 业务体积 | ≈0.67 MB | ✅ 701,131 B |
| 3 | 无排除路径混入 | 不出现 `botong25`、`demo`、`ReferenceDLL`、`document/data`、`build/`、`.vs/`、`.git/` | ✅ |
| 4 | 凭据 0 | 文件名/内容扫描无 `ghp_`、`PRIVATE KEY` | ✅ |
| 5 | 无大二进制 | 最大文件 `CAPP/src/Undistortion.cpp` 57 KB；无 DLL/EXE/PNG 二进制 | ✅ |
| 6 | 根自述 4 件 | `README.md` + `UPLOAD_RULES.md` + `.gitignore` + `.gitattributes` | ✅ |
| 7 | 分支/标签 | `cpu-v0` + 附注标签 `26452`；**不影响 `main` / `26453`** | ✅（推送后核对） |

### 已知但不阻断

- 本分支不含矩阵与测试图（960 MB + 2.4 GB），README 已注明“需自备”；
- `CSharpApp` 的 `ProjectReference` 指向 `CAPP\build\Undistortion.vcxproj`（CMake 产物，不入库），
  克隆后必须先跑 CMake，README 已用 ⚠️ 标注。
