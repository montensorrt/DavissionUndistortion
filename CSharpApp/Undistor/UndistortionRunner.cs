using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;

namespace Undistor.NativeAPI;

/// <summary>
/// Undistortion 主运行程序，参考 CAPP/test/test_undistortion.cpp 的完整处理流程
/// </summary>
public class UndistortionRunner
{
    // 配置常量（对应 C++ 中的配置）
    private static readonly string[] ImageExtensions = { ".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif" };

    /// <summary>
    /// 顺时针旋转90度
    /// </summary>
    private static Mat RotateClockwise90(Mat image)
    {
        Mat rotated = new Mat();
        Cv2.Rotate(image, rotated, RotateFlags.Rotate90Clockwise);
        return rotated;
    }

    /// <summary>
    /// 逆时针旋转90度
    /// </summary>
    private static Mat RotateCounterClockwise90(Mat image)
    {
        Mat rotated = new Mat();
        Cv2.Rotate(image, rotated, RotateFlags.Rotate90Counterclockwise);
        return rotated;
    }

    /// <summary>
    /// 初始化日志系统
    /// </summary>
    private static void InitializeLogger()
    {
        try
        {
            var logPath = System.Text.Encoding.UTF8.GetBytes("./logs/undistortion_csharp.log\0");
            var logConfig = new LoggerConfig_NativeAPI
            {
                logPath = new byte[256],
                level = 1, // DEBUG级别
                maxFileSize = 10 * 1024 * 1024,
                maxFiles = 5,
                enableConsole = 1,
                enableFile = 1
            };
            Array.Copy(logPath, logConfig.logPath, Math.Min(logPath.Length, 256));
            
            int logEc = Logger_NativeAPI.AOI_Logger_Configure(ref logConfig);
            if (logEc == 0)
            {
                Console.WriteLine("日志系统初始化完成");
            }
        }
        catch
        {
            Console.WriteLine("日志系统初始化失败，继续执行");
        }
    }

    /// <summary>
    /// 递归查找图像文件
    /// </summary>
    private static List<string> FindImageFiles(string rootPath)
    {
        var imageFiles = new List<string>();
        try
        {
            foreach (string ext in ImageExtensions)
            {
                imageFiles.AddRange(Directory.GetFiles(rootPath, "*" + ext, SearchOption.AllDirectories));
            }
        }
        catch (Exception e)
        {
            Console.WriteLine($"遍历目录时出错: {e.Message}");
        }
        return imageFiles;
    }

    /// <summary>
    /// 计算平均耗时（去除最大最小值）
    /// </summary>
    private static float CalculateAverageWithoutMinMax(List<float> times)
    {
        if (times.Count < 3)
        {
            return times.Count == 0 ? 0.0f : times.Average();
        }

        times.Sort();
        // 去除最小值和最大值
        var validTimes = times.Skip(1).Take(times.Count - 2);
        return validTimes.Any() ? (float)validTimes.Average() : 0.0f;
    }

    /// <summary>
    /// 完整的去畸变处理流程（对应 C++ 主处理流程）
    /// </summary>
    /// <param name="matrixPath">插值矩阵文件夹路径</param>
    /// <param name="inputImagePath">输入图像路径</param>
    /// <param name="outputImagePath">输出图像路径</param>
    /// <param name="imageWidth">图像宽度</param>
    /// <param name="imageHeight">图像高度</param>
    public static void Run(string matrixPath, string inputImagePath, string outputImagePath, 
        int imageWidth = 5120, int imageHeight = 4096)
    {
        Console.WriteLine("=== Undistortion C# Test Runner 开始执行 ===");
        
        // 检查输入
        if (!File.Exists(inputImagePath))
        {
            Console.WriteLine($"错误：输入图像文件不存在 - {inputImagePath}");
            return;
        }

        if (!Directory.Exists(matrixPath))
        {
            Console.WriteLine($"错误：矩阵文件夹不存在 - {matrixPath}");
            return;
        }

        try
        {
            // 1. 初始化日志系统
            InitializeLogger();

            // 2. 显示版本信息
            try
            {
                int ec = CommonTypes_NativeAPI.AOI_GetVersion(out int major, out int minor, out int patch);
                Console.WriteLine(ec == 0 ? $"API 版本: {major}.{minor}.{patch}" : "版本信息获取失败");
            }
            catch
            {
                Console.WriteLine("版本信息不可用");
            }

            // 3. 加载输入图像
            Console.WriteLine($"加载输入图像: {inputImagePath}");
            using var inputImage = new Mat(inputImagePath, ImreadModes.Grayscale);
            
            if (inputImage.Empty())
            {
                Console.WriteLine("错误：无法加载输入图像");
                return;
            }

            Console.WriteLine($"输入图像尺寸: {inputImage.Width} × {inputImage.Height}");

            // 4. 执行完整的处理流程（对应 C++ 的旋转 -> 去畸变 -> 反向旋转）
            Console.WriteLine("开始完整处理流程...");
            Console.WriteLine("=== 调试日志：图像尺寸追踪 ===");
            
            var totalStart = DateTime.Now;

            // 4.1 先顺时针旋转90度
            Console.WriteLine($"[步骤1] 原始图像尺寸: {inputImage.Width} × {inputImage.Height}");
            var rotateStart = DateTime.Now;
            using var rotatedImage = RotateClockwise90(inputImage);
            var rotateTime = (DateTime.Now - rotateStart).TotalMilliseconds;
            Console.WriteLine($"[步骤1] 顺时针旋转90度完成，耗时: {rotateTime:F2} ms");
            Console.WriteLine($"[步骤1] 旋转后图像尺寸: {rotatedImage.Width} × {rotatedImage.Height}");

            // 4.2 初始化去畸变处理器（对应C++的MATRIX_WIDTH=5120, MATRIX_HEIGHT=4096）
            Console.WriteLine($"[步骤2] C++理论尺寸: MATRIX_WIDTH=5120, MATRIX_HEIGHT=4096");
            Console.WriteLine($"[步骤2] 实际旋转后尺寸: {rotatedImage.Width} × {rotatedImage.Height}");
            Console.WriteLine($"[步骤2] 尺寸匹配检查: {(rotatedImage.Width == 5120 && rotatedImage.Height == 4096 ? "✅ 完全匹配" : "❌ 尺寸不匹配")}");
            using var processor = new UndistortionProcessor();
            bool initResult = processor.Initialize(matrixPath, 5120, 4096, 1); // 1=灰度图
            if (!initResult)
            {
                Console.WriteLine("错误：去畸变处理器初始化失败");
                return;
            }
            Console.WriteLine($"[步骤2] 去畸变处理器初始化完成 (对应C++配置)");

            // 4.3 去畸变处理
            Console.WriteLine($"[步骤3] 开始去畸变处理...");
            var undistortStart = DateTime.Now;
            using var undistortedImage = processor.ProcessImage(rotatedImage);
            if (undistortedImage == null)
            {
                Console.WriteLine("错误：去畸变处理失败");
                return;
            }
            var undistortTime = (DateTime.Now - undistortStart).TotalMilliseconds;
            Console.WriteLine($"[步骤3] 去畸变处理完成，耗时: {undistortTime:F2} ms");
            Console.WriteLine($"[步骤3] 去畸变后图像尺寸: {undistortedImage.Width} × {undistortedImage.Height}");

            // 4.4 逆时针旋转90度（恢复原始方向）
            Console.WriteLine($"[步骤4] 开始逆时针旋转90度（恢复原始方向）...");
            var rotateBackStart = DateTime.Now;
            using var finalImage = RotateCounterClockwise90(undistortedImage);
            var rotateBackTime = (DateTime.Now - rotateBackStart).TotalMilliseconds;
            Console.WriteLine($"[步骤4] 逆时针旋转完成，耗时: {rotateBackTime:F2} ms");
            Console.WriteLine($"[步骤4] 最终图像尺寸: {finalImage.Width} × {finalImage.Height}");

            var totalTime = (DateTime.Now - totalStart).TotalMilliseconds;
            Console.WriteLine($"总处理耗时: {totalTime:F2} ms");
            
            // 6. 保存输出图像
            Directory.CreateDirectory(Path.GetDirectoryName(outputImagePath) ?? ".");
            finalImage.SaveImage(outputImagePath);
            Console.WriteLine($"输出图像已保存: {outputImagePath}");
            Console.WriteLine($"最终图像尺寸: {finalImage.Width} × {finalImage.Height}");

            // 7. 刷新日志
            try
            {
                Logger_NativeAPI.AOI_Logger_Flush();
            }
            catch { }
            
            Console.WriteLine("=== 单张图像处理完成 ===");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"处理错误：{ex.Message}");
            Console.WriteLine($"堆栈跟踪：{ex.StackTrace}");
        }
    }

    /// <summary>
    /// 批量处理示例（对应 C++ 的递归目录处理和性能统计）
    /// </summary>
    /// <param name="matrixPath">插值矩阵文件夹路径</param>
    /// <param name="inputFolderPath">输入图像文件夹路径</param>
    /// <param name="outputFolderPath">输出图像文件夹路径</param>
    public static void RunBatch(string matrixPath, string inputFolderPath, string outputFolderPath)
    {
        Console.WriteLine("=== Undistortion Batch Runner 开始执行 ===");
        Console.WriteLine($"输入路径: {inputFolderPath}");
        Console.WriteLine($"矩阵路径: {matrixPath}");
        Console.WriteLine($"输出路径: {outputFolderPath}");

        if (!Directory.Exists(inputFolderPath))
        {
            Console.WriteLine($"错误：输入文件夹不存在 - {inputFolderPath}");
            return;
        }

        if (!Directory.Exists(matrixPath))
        {
            Console.WriteLine($"错误：矩阵文件夹不存在 - {matrixPath}");
            return;
        }

        try
        {
            // 1. 初始化日志系统
            InitializeLogger();

            // 2. 递归查找所有图像文件
            Console.WriteLine("递归查找图像文件...");
            var imageFiles = FindImageFiles(inputFolderPath);

            if (imageFiles.Count == 0)
            {
                Console.WriteLine("未找到图像文件");
                return;
            }

            Console.WriteLine($"找到 {imageFiles.Count} 个图像文件");

            // 3. 初始化去畸变处理器
            Console.WriteLine("=== 批量处理调试日志 ===");
            
            // 使用第一张图像确定尺寸
            using var firstImage = new Mat(imageFiles[0], ImreadModes.Grayscale);
            if (firstImage.Empty())
            {
                Console.WriteLine("无法加载第一张图像来确定尺寸");
                return;
            }

            Console.WriteLine($"[初始化] 第一张图像原始尺寸: {firstImage.Width} × {firstImage.Height}");
            using var firstRotated = RotateClockwise90(firstImage);
            Console.WriteLine($"[初始化] 第一张图像旋转后尺寸: {firstRotated.Width} × {firstRotated.Height}");
            Console.WriteLine($"[初始化] 理论初始化尺寸: 5120 × 4096 (硬编码)");
            
            using var processor = new UndistortionProcessor();
            bool batchInitResult = processor.Initialize(matrixPath, 5120, 4096, 1); // 1=灰度图
            if (!batchInitResult)
            {
                Console.WriteLine("错误：批量处理器初始化失败");
                return;
            }
            Console.WriteLine($"[初始化] 处理器初始化完成 (5120×4096)");

            // 4. 批量处理并统计性能
            Directory.CreateDirectory(outputFolderPath);
            
            var rotateClockwiseTimes = new List<float>();
            var undistortTimes = new List<float>();
            var rotateCounterTimes = new List<float>();
            
            int processedCount = 0;
            int failedCount = 0;
            var totalBatchStart = DateTime.Now;

            foreach (string imagePath in imageFiles)
            {
                try
                {
                    // 加载图像
                    using var inputImage = new Mat(imagePath, ImreadModes.Grayscale);
                    if (inputImage.Empty())
                    {
                        Console.WriteLine($"跳过无法加载的图像: {Path.GetFileName(imagePath)}");
                        failedCount++;
                        continue;
                    }

                    // 为第一张图像显示详细调试信息
                    bool isFirstImage = (processedCount == 0);
                    if (isFirstImage)
                    {
                        Console.WriteLine($"\n=== 第一张图像详细处理日志 ({Path.GetFileName(imagePath)}) ===");
                        Console.WriteLine($"[批量-步骤1] 原始图像尺寸: {inputImage.Width} × {inputImage.Height}");
                    }

                    // 5. 顺时针旋转90度并计时
                    var rotateStart = DateTime.Now;
                    using var rotatedImage = RotateClockwise90(inputImage);
                    rotateClockwiseTimes.Add((float)(DateTime.Now - rotateStart).TotalMilliseconds);
                    
                    if (isFirstImage)
                    {
                        Console.WriteLine($"[批量-步骤1] 旋转后图像尺寸: {rotatedImage.Width} × {rotatedImage.Height}");
                        Console.WriteLine($"[批量-步骤2] 处理器期望尺寸: 5120 × 4096");
                        Console.WriteLine($"[批量-步骤2] 尺寸匹配检查: {(rotatedImage.Width == 5120 && rotatedImage.Height == 4096 ? "✅ 匹配" : "❌ 不匹配")}");
                    }

                    // 6. 去畸变处理并计时
                    var undistortStart = DateTime.Now;
                    using var undistortedImage = processor.ProcessImage(rotatedImage);
                    undistortTimes.Add((float)(DateTime.Now - undistortStart).TotalMilliseconds);
                    
                    if (undistortedImage == null)
                    {
                        Console.WriteLine($"跳过去畸变失败的图像: {Path.GetFileName(imagePath)}");
                        failedCount++;
                        continue;
                    }
                    
                    if (isFirstImage)
                    {
                        Console.WriteLine($"[批量-步骤2] 去畸变处理完成，输出尺寸: {undistortedImage.Width} × {undistortedImage.Height}");
                    }

                    // 7. 逆时针旋转90度并计时
                    var rotateBackStart = DateTime.Now;
                    using var finalImage = RotateCounterClockwise90(undistortedImage);
                    rotateCounterTimes.Add((float)(DateTime.Now - rotateBackStart).TotalMilliseconds);
                    
                    if (isFirstImage)
                    {
                        Console.WriteLine($"[批量-步骤3] 最终图像尺寸: {finalImage.Width} × {finalImage.Height}");
                        Console.WriteLine($"=== 第一张图像处理完成 ===\n");
                    }

                    // 8. 保存结果（保持相对路径结构）
                    string relativePath = Path.GetRelativePath(inputFolderPath, imagePath);
                    string outputPath = Path.Combine(outputFolderPath, relativePath);
                    Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
                    
                    finalImage.SaveImage(outputPath);
                    
                    processedCount++;
                    
                    // 显示进度
                    if (processedCount % 10 == 0)
                    {
                        Console.WriteLine($"进度: {processedCount}/{imageFiles.Count}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"处理失败 {Path.GetFileName(imagePath)}: {ex.Message}");
                    failedCount++;
                }
            }

            var totalBatchTime = (DateTime.Now - totalBatchStart).TotalSeconds;

            // 9. 计算各步骤平均耗时（去除最大最小值，对应 C++ 逻辑）
            float avgRotateClockwise = CalculateAverageWithoutMinMax(rotateClockwiseTimes);
            float avgUndistort = CalculateAverageWithoutMinMax(undistortTimes);
            float avgRotateCounter = CalculateAverageWithoutMinMax(rotateCounterTimes);

            // 10. 性能统计报告（对应 C++ 的性能摘要）
            Console.WriteLine("\n=== 性能统计报告 ===");
            Console.WriteLine($"总图像数量: {imageFiles.Count}");
            Console.WriteLine($"成功处理: {processedCount}");
            Console.WriteLine($"处理失败: {failedCount}");
            Console.WriteLine($"总耗时: {totalBatchTime:F2} 秒");
            Console.WriteLine("\n各步骤平均耗时（去除最大最小值）:");
            Console.WriteLine($"  顺时针旋转 (90°): {avgRotateClockwise:F3} ms");
            Console.WriteLine($"  去畸变处理: {avgUndistort:F3} ms");
            Console.WriteLine($"  逆时针旋转 (90°): {avgRotateCounter:F3} ms");
            Console.WriteLine($"  总平均耗时: {(avgRotateClockwise + avgUndistort + avgRotateCounter):F3} ms");
            
            if (processedCount > 0)
            {
                Console.WriteLine($"平均处理速度: {totalBatchTime / processedCount:F2} 秒/图");
            }

            // 11. 刷新日志
            try
            {
                Logger_NativeAPI.AOI_Logger_Flush();
            }
            catch { }

            Console.WriteLine("=== 批量处理完成 ===");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"批量处理错误：{ex.Message}");
            Console.WriteLine($"堆栈跟踪：{ex.StackTrace}");
        }
    }

    /// <summary>
    /// 主入口点（完全对应 C++ main 函数逻辑）
    /// </summary>
    public static void Main(string[] args)
    {
        // 基于 CAPP/test/test_undistortion.cpp 的完整配置
        string matrixPath = @"E:\algrithm\project\2DAOI\feipai_input\undistort\2DAOI";
        string inputPath = @"E:\algrithm\project\2DAOI\feipai_input\product_2\1022\143323\raw_grouped";
        string outputPath = @"E:\algrithm\project\2DAOI\feipai_input\product_2\1022\143323\undistorted_csharp_alone";

        Console.WriteLine("=== Undistortion C# Test Runner ===");
        Console.WriteLine($"矩阵路径: {matrixPath}");
        Console.WriteLine($"输入路径: {inputPath}");
        Console.WriteLine($"输出路径: {outputPath}");
        Console.WriteLine();

        // 处理命令行参数
        if (args.Length >= 3)
        {
            matrixPath = args[0];
            inputPath = args[1];
            outputPath = args[2];
        }

        // 判断是单张图像还是目录处理
        if (File.Exists(inputPath))
        {
            // 单张图像处理
            Console.WriteLine("检测到单张图像，执行单图处理模式...");
            Run(matrixPath, inputPath, outputPath);
        }
        else if (Directory.Exists(inputPath))
        {
            // 批量目录处理（对应 C++ 的完整递归处理流程）
            Console.WriteLine("检测到目录，执行批量处理模式...");
            RunBatch(matrixPath, inputPath, outputPath);
        }
        else
        {
            Console.WriteLine("输入路径不存在，尝试单图处理模式...");
            // 如果路径不存在，尝试作为单图处理（可能是未来要创建的文件）
            Run(matrixPath, inputPath, outputPath);
        }
        
        Console.WriteLine("\n按任意键退出...");
        Console.ReadKey();
    }
}