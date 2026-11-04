using System.Text.Json;
using OpenCvSharp;
using Undistortion.Inference;
using Undistortion.Tools;

// ===================================================================
// Undistortion.Tools —— 验收/比对工具（工具项目，不进交付库）
//
//   Undistortion.Tools verify --raw <目录> --matrix <矩阵目录>
//                             [--ref <CPU 参考目录>] [--out <输出目录>]
//                             [--limit N] [--json <报告文件>] [--device 0]
//
//   verify : 批量去畸变；给了 --ref 就逐像素比对并判定 PASS/FAIL，
//            不给就只跑一遍看耗时。--out 可选，落盘去畸变结果。
//
// 尺寸取自首图（必须与矩阵一致，原生会校验），不读配置文件、不写死 5120×4096。
// 判定阈值（与历史验收口径一致）：max|d| <= 1、差异比例 <= 1e-5、≥2 差异 == 0、PSNR >= 60 dB
// ===================================================================

const int MaxAbsDiffLimit = 1;
const double DiffRatioLimit = 1e-5;
const long CountGe2Limit = 0;
const double PsnrLimit = 60.0;

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

string command = args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal)
    ? args[0]
    : "verify";

if (command != "verify")
{
    Console.WriteLine("用法: Undistortion.Tools verify --raw <目录> [--matrix <矩阵目录>] " +
                      "[--ref <目录>] [--out <目录>] [--limit N] [--json <文件>] [--device 0]");
    Console.WriteLine("      --matrix 省略时用 <exe目录>\\botong25");
    return 2;
}

string? rawDir = Arg(args, "--raw");
if (rawDir is null || !Directory.Exists(rawDir))
{
    Console.Error.WriteLine("缺少 --raw 目录，或目录不存在");
    return 2;
}

// 默认矩阵位置：exe 同级 botong25（与 Undistortion.Test 一致）
string matrixDir = ResolveFromExeDir(Arg(args, "--matrix") ?? "botong25");
if (!Directory.Exists(matrixDir))
{
    Console.Error.WriteLine($"矩阵目录不存在: {matrixDir}");
    Console.Error.WriteLine("把 botong25（12 个 .bin）放到 exe 同级目录，或用 --matrix 指定绝对路径。");
    return 2;
}

string? refDir = Arg(args, "--ref");
string? outDir = Arg(args, "--out");
string? jsonPath = Arg(args, "--json");
int limit = int.TryParse(Arg(args, "--limit") ?? "0", out int n) ? n : 0;
int device = int.TryParse(Arg(args, "--device") ?? "0", out int deviceArg) ? deviceArg : 0;

string[] files = Directory.GetFiles(rawDir, "*.bmp")
    .OrderBy(p => Path.GetFileName(p), StringComparer.Ordinal)
    .ToArray();
if (limit > 0 && files.Length > limit) files = files[..limit];
if (files.Length == 0)
{
    Console.Error.WriteLine($"目录没有 bmp: {rawDir}");
    return 3;
}

// 尺寸取自首图
using Mat probe = Cv2.ImRead(files[0], ImreadModes.Grayscale);
if (probe.Empty())
{
    Console.Error.WriteLine($"读图失败: {files[0]}");
    return 3;
}

var config = new UndistortionConfig
{
    MatrixDir = matrixDir,
    Width = probe.Width,
    Height = probe.Height,
    DeviceId = device,
};

using var undistorter = new Undistorter(config);
Console.WriteLine($"[init] {undistorter.Info}");

using var dst = new Mat(config.Height, config.Width, MatType.CV_8UC1);
var rows = new List<(string Name, ImageDiff? Diff, double TotalMs)>();

foreach (string path in files)
{
    string name = Path.GetFileName(path);
    using Mat src = Cv2.ImRead(path, ImreadModes.Grayscale);
    if (src.Empty())
    {
        Console.Error.WriteLine($"读图失败: {path}");
        return 3;
    }

    undistorter.Process(src, dst);

    ImageDiff? diff = null;
    if (refDir is not null)
    {
        string refPath = Path.Combine(refDir, name);
        using Mat reference = Cv2.ImRead(refPath, ImreadModes.Grayscale);
        if (reference.Empty())
        {
            Console.Error.WriteLine($"参考图缺失: {refPath}");
            return 3;
        }
        diff = ImageComparer.Compare(dst, reference);
    }

    if (outDir is not null)
    {
        Directory.CreateDirectory(outDir);
        Cv2.ImWrite(Path.Combine(outDir, name), dst);
    }

    rows.Add((name, diff, undistorter.LastTiming.TotalMs));

    Console.WriteLine(diff is null
        ? $"[run] {name,-16} total={undistorter.LastTiming.TotalMs,7:F3} ms"
        : $"[run] {name,-16} diff={diff.DiffPixels,7} max={diff.MaxAbsDiff} ge2={diff.CountGe2} " +
          $"psnr={diff.PsnrDb,7:F2} dB total={undistorter.LastTiming.TotalMs,7:F3} ms");
}

double meanTotal = rows.Average(r => r.TotalMs);
Console.WriteLine($"\n[timing] 单图 mean={meanTotal:F3} ms ({1000.0 / meanTotal:F1} 张/s)");

bool? passed = null;
long totalDiff = 0, totalPixels = 0, totalGe2 = 0;
int maxAbs = 0;
double psnrMin = 999.0;

if (refDir is not null)
{
    foreach (var row in rows)
    {
        ImageDiff d = row.Diff!;
        totalDiff += d.DiffPixels;
        totalGe2 += d.CountGe2;
        maxAbs = Math.Max(maxAbs, d.MaxAbsDiff);
        psnrMin = Math.Min(psnrMin, d.PsnrDb);
    }
    totalPixels = (long)config.Width * config.Height * rows.Count;
    double diffRatio = totalPixels > 0 ? (double)totalDiff / totalPixels : 0.0;

    passed = maxAbs <= MaxAbsDiffLimit &&
             diffRatio <= DiffRatioLimit &&
             totalGe2 == CountGe2Limit &&
             psnrMin >= PsnrLimit;

    Console.WriteLine($"\n[accept] {(passed.Value ? "PASS" : "FAIL")}  " +
                      $"diff={totalDiff}/{totalPixels} ({diffRatio:E3}) " +
                      $"max={maxAbs} ge2={totalGe2} psnr_min={psnrMin:F2} dB");
}

if (jsonPath is not null)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(jsonPath))!);
    var report = new
    {
        meta = new
        {
            tool = "Undistortion.Tools/verify",
            timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            config = config.ToString(),
            raw_dir = Path.GetFullPath(rawDir),
            ref_dir = refDir is null ? null : Path.GetFullPath(refDir),
            images = rows.Count,
            width = config.Width,
            height = config.Height,
        },
        env = new
        {
            gpu = undistorter.Info.GpuName,
            sm_count = undistorter.Info.SmCount,
            tensorrt = undistorter.Info.TensorRtVersion,
            staging = undistorter.Info.PinnedStaging ? "pinned" : "pageable",
            table_bytes = undistorter.Info.TableBytes,
            device_bytes = undistorter.Info.DeviceBytes,
            init_total_ms = Math.Round(undistorter.Info.InitTotalMs, 1),
        },
        acceptance = new
        {
            passed,
            criteria = new { max_abs_diff = MaxAbsDiffLimit, diff_ratio = DiffRatioLimit, count_ge2 = CountGe2Limit, psnr_db = PsnrLimit },
            totals = refDir is null ? null : new
            {
                diff_pixels = totalDiff,
                pixels = totalPixels,
                max_abs_diff = maxAbs,
                count_ge2 = totalGe2,
                psnr_min_db = Math.Round(psnrMin, 3),
            },
        },
        timing = new { mean_total_ms = Math.Round(meanTotal, 3), images_per_second = Math.Round(meanTotal > 0 ? 1000.0 / meanTotal : 0, 1) },
        images = rows.Select(r => new
        {
            name = r.Name,
            diff_pixels = r.Diff?.DiffPixels,
            max_abs_diff = r.Diff?.MaxAbsDiff,
            count_ge2 = r.Diff?.CountGe2,
            psnr_db = r.Diff is null ? (double?)null : Math.Round(r.Diff.PsnrDb, 3),
            total_ms = Math.Round(r.TotalMs, 3),
        }),
    };

    File.WriteAllText(jsonPath, JsonSerializer.Serialize(report, new JsonSerializerOptions
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    }));
    Console.WriteLine($"[json] {jsonPath}");
}

return passed == false ? 1 : 0;
