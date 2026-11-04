using OpenCvSharp;
using Undistortion.Inference;

// ===================================================================
// Undistortion.Test —— 最小调用链（加载指定图像 → 初始化 → 执行 → 释放）
//
//   Undistortion.Test --image <图> [--matrix <矩阵目录>] [--device 0] [--out <bmp>]
//
//   1) 读图：尺寸直接取自图像（不写死 5120×4096，也不读配置文件）
//   2) 构造 UndistortionConfig 结构体 —— 初始化参数只有这一个载体
//   3) new Undistorter(config) 初始化 → Process(src) 执行 → 写盘
//   4) using 退出即释放（Dispose → undist_release）
//
// 矩阵目录约定（部署友好）：
//   --matrix 可省略 → 默认使用 <exe 所在目录>\botong25；
//   给了相对名（如 botong25）→ 同样按 <exe 所在目录> 解析，**不看当前工作目录**，
//   所以快捷方式 / 服务 / 计划任务从任何 CWD 启动都能找到。
// ===================================================================

string? Arg(string[] argv, string key)
{
    for (int i = 0; i < argv.Length - 1; i++)
        if (argv[i] == key) return argv[i + 1];
    return null;
}

// 相对路径按 exe 目录解析（而不是进程的当前工作目录）
static string ResolveFromExeDir(string path) =>
    Path.IsPathRooted(path)
        ? Path.GetFullPath(path)
        : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, path));

string? imagePath = Arg(args, "--image");
if (imagePath is null)
{
    Console.WriteLine("用法: Undistortion.Test --image <图> [--matrix <目录>] [--device 0] [--out <bmp>]");
    Console.WriteLine("      --matrix 省略时用 <exe目录>\\botong25");
    return 2;
}

// 默认矩阵位置：exe 同级 botong25（放置方式见 document/UNDIST-107）
string matrixDir = ResolveFromExeDir(Arg(args, "--matrix") ?? "botong25");
if (!Directory.Exists(matrixDir))
{
    Console.Error.WriteLine($"矩阵目录不存在: {matrixDir}");
    Console.Error.WriteLine("把 botong25（12 个 .bin）放到 exe 同级目录，或用 --matrix 指定绝对路径。");
    return 2;
}

// 1) 加载指定图像
using Mat src = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
if (src.Empty())
{
    Console.Error.WriteLine($"读图失败: {imagePath}");
    return 3;
}

// 2) 初始化参数（结构体就是全部配置）
var config = new UndistortionConfig
{
    MatrixDir = matrixDir,
    Width = src.Width,
    Height = src.Height,
    DeviceId = int.TryParse(Arg(args, "--device") ?? "0", out int device) ? device : 0,
};

// 3) 初始化 → 执行 → 4) 离开作用域释放
using var undistorter = new Undistorter(config);
using Mat dst = undistorter.Process(src);

string outPath = Arg(args, "--out")
    ?? Path.Combine("output", Path.GetFileNameWithoutExtension(imagePath) + ".bmp");
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
if (!Cv2.ImWrite(outPath, dst))
{
    Console.Error.WriteLine($"写图失败: {outPath}");
    return 4;
}

Console.WriteLine($"in    : {imagePath} ({src.Width}x{src.Height}, {src.Type()})");
Console.WriteLine($"matrix: {matrixDir}");
Console.WriteLine($"out   : {outPath} ({dst.Width}x{dst.Height})");
Console.WriteLine($"info  : {undistorter.Info}");
Console.WriteLine($"timing: {undistorter.LastTiming}");
return 0;
