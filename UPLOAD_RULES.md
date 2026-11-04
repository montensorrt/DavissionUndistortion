# 规则说明

本文件是 `montensorrt/DavissionUndistortion` 仓库 **`main` 分支（GPU 版本）** 的上传边界与核验口径。
基线来自 `Algorithms\masktoolV0\document\MaskTool_上传GitHub规则与逻辑_V2.html`
（GOLDENLINE 白名单机制 + MaskTool 规则 1~6），按本工程目录结构做最小必要替换。

| 项 | 值 |
|---|---|
| 真值源 | `<工作区>\Algorithms\undistortion`（**不是** git 仓库，单向同步） |
| 目标仓库 | `https://github.com/montensorrt/DavissionUndistortion.git` |
| 目标分支 | `main`（**GPU 版本**：TensorRT + CUDA 推理；CPU 版本在 `undistortion_standaloneV0`，不入本仓库） |
| 目标标签 | `26453`（附注标签，用户指定，不套 V0 的“分支名+版本号”前缀规则） |
| 同步方向 | **单向**：工程目录 → 仓库。改代码改工程目录，别改仓库后指望下次同步保留 |
| 仓库根 | 对应 `Algorithms\undistortion\` 这一层（不是 `csharp\`） |

---

## 规则 1 —— 屏蔽路径（按目录名匹配，任何层级生效）

遍历时直接剪枝，不进入以下目录：

```text
trt/build/            # CMake/Ninja 产物 + 引擎缓存（本机实测 29 文件 / 8 MB）
trt/build-*/  trt/out/
csharp/**/bin/  csharp/**/obj/  csharp/**/output/  csharp/.vs/
document/data/preview/            # 大图预览（实测 1 文件 / 20 MB）
bin/  obj/  .vs/  out/  dist/  build/  publish/  _out/  _logs/  __pycache__/
node_modules/  .mimocode/  .zcode/  .tmp_strip/  .vscode/  tu/
```

| 目录 | 文件数 | 体积 | 理由 |
|---|---:|---:|---|
| `trt/build/` | 29 | 8 MB | 构建产物 + 引擎缓存（`.engine` 绑定 GPU 架构与 TRT 版本，可再生） |
| `document/data/preview/` | 1 | 20 MB | 预览大图，属运行期产物 |
| `csharp/**/bin`、`obj`、`output` | 0 | 0 | 交付前已清理；`.gitignore` 仍保留规则防回归 |

> **注意**：`obj/` 会被 .NET 还原/构建**自动写回**。提交前必须用显式文件清单对拍，
> 不能凭“我刚才看过了”。

---

## 规则 2 —— 工程根白名单

`Algorithms\undistortion` 根下只取这 7 类：

| 条目 | 取什么 | 理由 |
|---|---|---|
| `README.md` | 工程自述（本次加“仓库说明 + GPU 版本”头部） | 第一入口 |
| `.gitignore` | 排除规则 | 第二道保险 |
| `csharp/` | 3 个 C# 项目 + `thirdpart/undist_api.dll` + `document/`（3 份 HTML） | 交付层 |
| `trt/` | `CMakeLists.txt` + `include/` 3 + `src/` 3 + `tests/` 1 | 原生库源码（**不含 build/**） |
| `scripts/` | 5 个 `.ps1` + 1 个 `.py` | 构建/测试/验收/批处理 |
| `document/` | 101~105 HTML + `assets/` 2 图 + `data/` 3 份实测数据 | 设计与验收产出 |
| `prompt/` | `deploy_api.md`、`安装包.md`、`方案分析梳理.md`、`测试统计耗时.md` | 需求原文留档 |

### 规则 3 —— 逐类白名单（等价于逐文件）

| 路径 | 文件 | 说明 |
|---|---|---|
| `csharp/Undistortion.Inference/` | `Undistorter.cs`、`UndistortionConfig.cs`、`UndistortionInfo.cs`、`UndistortionException.cs`、`NativeMethods.cs`、`Undistortion.Inference.csproj` | 5 源 + 1 工程 |
| `csharp/Undistortion.Test/` | `Program.cs`、`Undistortion.Test.csproj` | 最小调用链 |
| `csharp/Undistortion.Tools/` | `Program.cs`、`ImageComparer.cs`、`Undistortion.Tools.csproj` | 验收工具 |
| `csharp/thirdpart/` | `undist_api.dll`（88,064 B）、`README.md` | **交付快照**，客户端不编译也能跑 |
| `csharp/document/` | `UNDIST-106/107/108-*.html` | C# 侧文档 |
| `csharp/` | `Directory.Build.props`、`Undistortion.sln` | 公共属性 + 解决方案 |
| `trt/include/` | `undist_api.h`、`undist_engine.h`、`undist_tables.h` | C ABI + 引擎 + 表 |
| `trt/src/` | `undist_api.cpp`、`undist_engine.cpp`、`undist_tables.cpp` | 实现 |
| `trt/tests/` | `test_undist_api.cpp` | 原生四口径验收程序 |
| `scripts/` | `_env.ps1`、`build_native.ps1`、`link_matrix.ps1`、`run_test.ps1`、`run_verify.ps1`、`run_acceptance.ps1`、`undist_batch.py` | 7 个 |
| `document/` | `UNDIST-101~105-*.html`、`assets/*.jpg` ×2、`data/*.json|csv` ×3 | 11 个 |

**为什么 `thirdpart/undist_api.dll` 要入库**：客户端没有 VS/CMake/CUDA SDK，
靠这一个 88 KB 的文件就能跑；`trt/build/` 里的同名产物反而不入库（可重建、且含缓存）。

---

## 规则 4 —— 其余一律不传

| 条目 | 体积 | 排除理由 |
|---|---:|---|
| `trt/build/`（含 `cache/*.engine`） | 8 MB | 规则 1：构建产物 + 引擎缓存 |
| `document/data/preview/sample_pinned.bmp` | 20 MB | 规则 1：预览大图 |
| `prompt/github.md` | 784 B | 规则 5：本次上传指令，正文含**凭据文件路径**，不入库 |
| `csharp/**/bin|obj|output` | — | 规则 1：构建产物 |
| 工作区级 `dataset/`、`document/`、`github/` | — | 不在本工程根下，天然不遍历 |

---

## 规则 5 —— 凭据红线（硬规则，无例外）

- 令牌文件位于 **工作区级** `github\person-access-token.md`，**不在本工程目录内**，天然不被遍历；
- 令牌**只允许**出现在推送命令的进程参数里（`http.extraheader`），用完即失效于进程；
  **不得**写入 `.git/config`、不得写进任何文件、不得进提交历史；
- 推送使用 `-c credential.helper=` 关闭凭据缓存，避免被 Windows 凭据管理器留存；
- 推完自检：`git remote -v` 不应含令牌；`git config -l | findstr -i token` 应为空；
- 本工程 `.gitignore` 已包含 `*access-token*`、`*access_token*`、`*secret*`、`.env*` 作为第二道保险。

---

## 规则 6 —— 变更记录纪律

- 每次同步在仓库根 `README.md` 的“快照统计”块里更新文件数/体积/时间；
- 标签只在**内容冻结**时打；同一标签不重复指向不同提交（本次为仓库首次上传，`26453` 唯一）；
- 修复记录写进工程 `document/` 的文档，不要只写在提交信息里。

---

## 上传逻辑（五步）

```text
[1/5] 白名单枚举     按规则 2/3 列出文件（本次 50 个业务文件）
[2/5] 逐文件比对     全新仓库 → 全部为“新增”
[3/5] 敏感信息体检   凭据 0；本机绝对路径仅出现在 document/data/*.json 的实测记录里（见下）
[4/5] 逐字节复制     复制到临时目录（不放工程目录，工程目录不建 .git）
[5/5] 写自述文件     README.md（仓库说明 + GPU 版本）/ UPLOAD_RULES.md / .gitattributes
```

推送（令牌只在第 2 条命令里出现，不落盘）：

```powershell
# 0) 在临时目录准备发布仓库（工程目录不建 .git）
git init -b main
git add -A
git commit -m "feat(gpu): Undistortion GPU 版本快照（TRT 推理 + Mat-only C# 封装）"
git remote add origin https://github.com/montensorrt/DavissionUndistortion.git

# 1) 推送主分支
git -c credential.helper= -c http.extraheader="AUTHORIZATION: basic <base64(x-access-token:TOKEN)>" push -u origin main

# 2) 打附注标签（标签消息说明 GPU 版本）
git tag -a 26453 -m "Undistortion 快照 · 26453（GPU 版本 / TensorRT 10.14 + CUDA 12.8）"
git -c credential.helper= -c http.extraheader="AUTHORIZATION: basic <同上>" push origin 26453
```

---

## 核验清单（提交前 / 推送后）

| # | 检查点 | 判据 | 本次实测 |
|---|---|---|---|
| 1 | 业务文件数 | 50 | ✅ 50 |
| 2 | 业务体积 | ≈1.9 MB | ✅ 约 1.9 MB（52 文件） |
| 3 | 无排除路径混入 | `git status --porcelain` 不出现 `trt/build`、`preview`、`bin/`、`obj/` | ✅ |
| 4 | 凭据 0 | 文件名/内容扫描无 `ghp_`、`PRIVATE KEY`、`token=` | ✅ |
| 5 | 无二进制大件 | 最大文件 `undist_api.dll` 88,064 B、`*.jpg` 755,137 B | ✅ |
| 6 | 仓库根自述 | `README.md` + `UPLOAD_RULES.md` + `.gitignore` + `.gitattributes` | ✅ 4 件 |
| 7 | 分支/标签 | `main` + 附注标签 `26453` | ✅（推送后核对） |

### 已知但不阻断的事项

- `document/data/*.json` 的实测记录里含**本机绝对路径**（`E:\algrithm\project\botong\davissionNew\...`）。
  这是测量证据的一部分（记录当时的数据目录），不含凭据；如需脱敏可改为相对路径后重新同步。
- 文档文件名保留历史编号 `UndistortionV1`（101~104），与目录名 `undistortion` 不一致，属历史命名，不影响构建。
