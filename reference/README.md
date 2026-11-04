# reference/ —— 只读参考实现归档

本目录**不参与任何构建**。`CAPP/CMakeLists.txt` 的源文件收集范围是
`${CMAKE_SOURCE_DIR}/src/*.cpp`（即 `CAPP/src/`），不会扫描到这里，
因此放在这里的文件不会被编译、也不会引入重复符号。

## 内容

| 文件 | 来源 | 说明 |
|---|---|---|
| `Undistortion_raw.cpp` | 原 `CPPAPP/src/Undistortion_raw.cpp`（20,804 B） | 去畸变算法的**原始 OpenCV `cv::Mat` 版实现** |

## 为什么保留它

`CAPP/src/Undistortion.cpp` 的 SIMD 版本是从这个 `cv::Mat` 版本重写优化的。
其中「**全有或全无（all-or-nothing）越界策略**」这一条关键语义的出处就是它：

> 4 个插值点只要有一个越界，整个像素的 4 个索引就全部改指向哑像素 `N`（值为 0），
> 权重不清零，于是结果为 `0 × Σw = 0`（纯黑）。
> `CAPP/src/Undistortion.cpp:503` 处的注释即指向本文件。

实测该策略造成 **350,805 / 20,971,520 = 1.6728%** 的像素恒为 0，
是芯片侧黑边比例的直接来源，任何移植（GPU / TRT / 其它平台）都必须逐位复刻。
保留原始实现便于日后语义核对与争议追溯。

## 为什么不能把它放回 `src/`

它是**另一个完整的 `ImageUndistorter` 实现**，与 `Undistortion.cpp` 并存时必然失败：

1. `CommonTypes.h` 中 `UndistortionConfig::isValid()/reset()` 已是 `inline` 定义，
   本文件又给出实体 → `error C2084`（重复定义）。
2. 本文件使用了 `LOG_DEBUGF`，而现行 `Logger.h` 只提供
   `LOG_DEBUG(module, fmt, ...)` 形式 → `error C3861`（找不到标识符）。
3. 即便修好前两项，本文件完整实现了 `ImageUndistorter` 的构造函数、析构函数、
   移动构造/赋值、`setConfig`、`isConfigured`、`infer`，
   与 `Undistortion.cpp` 逐一重复 → 链接期 `LNK2005` 重复符号。

以上前两项已在 2026-10-26 实测复现（共 6 个 error），第三项为静态分析结论。

## 历史

原 `CPPAPP/` 目录是同一份代码的同源分叉，与 `CAPP/` 逐文件比对结果为
**22 个文件字节相同 / 5 个不同 / 各 1 个独有**；
其构建脚本产不出 C 接口 DLL，且因上述原因从未编译成功。
2026-10-26 已删除整个 `CPPAPP/`，仅将本文件归档至此。
