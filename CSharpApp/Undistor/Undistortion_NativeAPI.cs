using System;
using System.Runtime.InteropServices;

namespace Undistor.NativeAPI;

/// <summary>
/// 去畸变配置（NativeAPI 版本，对应底层 AOI_UndistortionConfig 结构）
/// <para>关键字段说明:</para>
/// <list type="bullet">
/// <item><description><c>matrixPath</c>: 插值矩阵文件夹路径（包含 U_1~4.bin, V_1~4.bin, A1~4.bin）</description></item>
/// <item><description><c>matrixWidth/matrixHeight</c>: 权重矩阵的宽度和高度（必须与输入图像尺寸一致）</description></item>
/// <item><description><c>outputChannels</c>: 输出通道数（1=灰度图像）</description></item>
/// </list>
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct UndistortionConfig_NativeAPI
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 512)]
    public byte[] matrixPath;   // 插值矩阵文件夹路径（UTF-8编码）
    public int matrixWidth;     // 权重矩阵宽度
    public int matrixHeight;    // 权重矩阵高度
    public int outputChannels;  // 输出通道数（1=灰度，3=彩色）
}

/// <summary>
/// 去畸变结果（对应 AOI_BaseResult 结构）
/// <para><c>errorCode</c> 标志为 0 表示算法成功，非0表示失败</para>
/// <para><c>errorMessage</c> 为 UTF-8 以 0 结尾的错误描述，失败时可解析读取</para>
/// <para><c>processingTime</c> 记录处理耗时（毫秒）</para>
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct UndistortionResult_NativeAPI
{
    public int errorCode;     // 0=成功，非0=失败
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] errorMessage; // UTF-8 错误消息，以 0 结尾
    public float processingTime; // 处理耗时（毫秒）
}

/// <summary>
/// Undistortion 原生接口（对应 AOI_Undistortion_* C API 函数）
/// <para>通常不直接在业务代码中调用，推荐通过 <c>UndistortionProcessor</c> 进行高级封装</para>
/// <para>手动调用示例（不推荐，仅供参考）:</para>
/// <code language="csharp">
/// IntPtr h = Undistortion_NativeAPI.AOI_Undistortion_Create();
/// var cfg = new UndistortionConfig_NativeAPI 
/// { 
///     matrixPath = Encoding.UTF8.GetBytes("./matrices\0"), 
///     matrixWidth = 5120, 
///     matrixHeight = 4096, 
///     outputChannels = 1 
/// };
/// int ec = Undistortion_NativeAPI.AOI_Undistortion_SetConfig(h, ref cfg);
/// if (ec != 0 || Undistortion_NativeAPI.AOI_Undistortion_IsConfigured(h) == 0) 
///     throw new Exception("配置失败");
/// // 准备输入输出缓冲区...
/// // 调用 AOI_Undistortion_Infer(h, inputPtr, width, height, outputPtr, out result);
/// Undistortion_NativeAPI.AOI_Undistortion_Destroy(h);
/// </code>
/// </summary>
public static class Undistortion_NativeAPI
{
#if DEBUG
    private const string DLL_NAME = "Undistortion_d.dll";
#else
    private const string DLL_NAME = "Undistortion.dll";
#endif
    private const CallingConvention CC = CallingConvention.Cdecl;

    /// <summary>
    /// 创建原生 Undistortion 实例
    /// <para>返回的指针需要使用结束后调用 <see cref="AOI_Undistortion_Destroy"/> 释放，建议用 SafeHandle 封装</para>
    /// <para>失败时返回 <see cref="IntPtr.Zero"/></para>
    /// </summary>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern IntPtr AOI_Undistortion_Create();

    /// <summary>
    /// 销毁原生 Undistortion 实例，释放底层资源
    /// <para>调用后指针即失效，不能再次使用</para>
    /// </summary>
    /// <param name="undistorter">由 <see cref="AOI_Undistortion_Create"/> 获取的实例指针</param>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern void AOI_Undistortion_Destroy(IntPtr undistorter);

    /// <summary>
    /// 设置去畸变配置。需要在调用 <see cref="AOI_Undistortion_IsConfigured"/> 和 <see cref="AOI_Undistortion_Infer"/> 前执行
    /// <para>返回值为错误码（0=成功，非0表示无效配置或其他配置失败）</para>
    /// <para>注意：配置成功不等于完全就绪，建议使用 <see cref="AOI_Undistortion_IsConfigured"/> 验证</para>
    /// </summary>
    /// <param name="undistorter">实例指针</param>
    /// <param name="config">去畸变配置结构（引用传递，使用 ref）</param>
    /// <returns>错误码：0 成功，非0表示失败</returns>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern int AOI_Undistortion_SetConfig(IntPtr undistorter, ref UndistortionConfig_NativeAPI config);

    /// <summary>
    /// 检查实例是否已正确配置。通常在 <see cref="AOI_Undistortion_SetConfig"/> 调用后检查
    /// <para>返回 1 表示已配置，0 表示未配置或配置失败</para>
    /// <para>即使SetConfig 返回 0 ，仍建议调用此方法确认</para>
    /// </summary>
    /// <param name="undistorter">实例指针</param>
    /// <returns>1=已配置，0=未配置</returns>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern int AOI_Undistortion_IsConfigured(IntPtr undistorter);

    /// <summary>
    /// 执行图像去畸变处理
    /// <para>输入输出图像数据必须为连续内存，大小 = width × height × channels</para>
    /// <para>函数执行流程：</para>
    /// <list type="number">
    /// <item><description>验证输入参数有效性（指针非空、尺寸匹配等）</description></item>
    /// <item><description>执行SIMD优化的去畸变算法（AVX2/FMA加速）</description></item>
    /// <item><description>填充结果结构体，包含错误信息和处理耗时</description></item>
    /// </list>
    /// </summary>
    /// <param name="undistorter">已正确配置的实例指针</param>
    /// <param name="inputImage">输入图像数据指针</param>
    /// <param name="width">输入图像宽度</param>
    /// <param name="height">输入图像高度</param>
    /// <param name="outputImage">输出图像数据指针（预分配足够内存）</param>
    /// <param name="result">处理结果结构体（包含成功标志、错误信息、耗时）</param>
    /// <returns>错误码：0=调用执行成功，非0=失败（具体原因参考错误码定义）</returns>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern int AOI_Undistortion_Infer(
        IntPtr undistorter,
        IntPtr inputImage,
        int width,
        int height,
        IntPtr outputImage,
        out UndistortionResult_NativeAPI result);
}