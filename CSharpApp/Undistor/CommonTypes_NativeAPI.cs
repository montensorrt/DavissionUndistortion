using System;
using System.Runtime.InteropServices;

namespace Undistor.NativeAPI;

/// <summary>
/// 基础结果类（对应 BaseResult）
/// <para>通用结果结构，用于返回处理状态和错误信息</para>
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct BaseResult_NativeAPI
{
    public int errorCode;  // 错误码：0=成功，非0表示失败
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] errorMessage;  // UTF-8 错误消息，以 0 结尾
    public float processingTime;  // 处理耗时（毫秒）
}

/// <summary>
/// 日志配置结构（对应 LogConfig）
/// <para>用于配置日志系统的输出路径、级别等参数</para>
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct LogConfig_NativeAPI
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] logPath;  // 日志文件路径（UTF-8）
    public int level;       // 日志级别：0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR
    public int maxFileSize; // 最大文件大小（字节）
    public int maxFiles;    // 最大文件数量
    public int enableConsole; // 启用控制台输出：1=启用，0=禁用
    public int enableFile;    // 启用文件输出：1=启用，0=禁用
}

/// <summary>
/// CommonTypes 原生接口（对应 AOI_GetVersion 等通用函数）
/// <para>提供版本信息查询等通用功能</para>
/// </summary>
public static class CommonTypes_NativeAPI
{
#if DEBUG
    private const string DLL_NAME = "Undistortion_d.dll";
#else
    private const string DLL_NAME = "Undistortion.dll";
#endif
    private const CallingConvention CC = CallingConvention.Cdecl;

    /// <summary>
    /// 获取API版本信息
    /// <para>返回主版本号、次版本号和补丁版本号</para>
    /// </summary>
    /// <param name="major">主版本号</param>
    /// <param name="minor">次版本号</param>
    /// <param name="patch">补丁版本号</param>
    /// <returns>返回码：0=成功，非0表示失败</returns>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern int AOI_GetVersion(out int major, out int minor, out int patch);
}