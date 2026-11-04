using System;
using System.Runtime.InteropServices;

namespace Undistor.NativeAPI;

/// <summary>
/// 日志配置结构（原生API版本，对应底层 AOI_LogConfig 结构）
/// <para>关键字段说明:</para>
/// <list type="bullet">
/// <item><description><c>logPath</c>: 日志文件存储路径（UTF-8编码）</description></item>
/// <item><description><c>level</c>: 日志级别（0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR）</description></item>
/// <item><description><c>maxFileSize</c>: 单个日志文件最大大小（字节）</description></item>
/// <item><description><c>maxFiles</c>: 保留的日志文件最大数量</description></item>
/// <item><description><c>enableConsole/enableFile</c>: 启用控制台/文件输出（1=启用，0=禁用）</description></item>
/// </list>
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct LoggerConfig_NativeAPI
{
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
    public byte[] logPath;
    public int level;
    public int maxFileSize;
    public int maxFiles;
    public int enableConsole;
    public int enableFile;
}

/// <summary>
/// Logger 原生接口（对应 AOI_Logger_* C API 函数）
/// <para>提供日志系统的配置和管理功能</para>
/// <para>推荐使用流程：</para>
/// <code language="csharp">
/// var cfg = new LoggerConfig_NativeAPI 
/// { 
///     logPath = Encoding.UTF8.GetBytes("./logs/app.log\0"), 
///     level = 2, // INFO
///     maxFileSize = 10 * 1024 * 1024, 
///     maxFiles = 5, 
///     enableConsole = 1, 
///     enableFile = 1 
/// };
/// int ec = Logger_NativeAPI.AOI_Logger_Configure(ref cfg);
/// if (ec != 0) throw new Exception("日志配置失败");
/// </code>
/// </summary>
public static class Logger_NativeAPI
{
#if DEBUG
    private const string DLL_NAME = "Undistortion_d.dll";
#else
    private const string DLL_NAME = "Undistortion.dll";
#endif
    private const CallingConvention CC = CallingConvention.Cdecl;

    /// <summary>
    /// 配置日志系统参数
    /// <para>必须在使用其他日志功能前调用此函数进行初始化</para>
    /// </summary>
    /// <param name="config">日志配置结构（引用传递，避免数据拷贝）</param>
    /// <returns>返回码：0=成功，非0表示失败</returns>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern int AOI_Logger_Configure(ref LoggerConfig_NativeAPI config);

    /// <summary>
    /// 检查日志系统是否已配置
    /// <para>返回 1 表示已配置，0 表示未配置或配置失败</para>
    /// </summary>
    /// <returns>1=已配置，0=未配置</returns>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern int AOI_Logger_IsConfigured();

    /// <summary>
    /// 刷新日志缓冲区，确保所有日志内容写入文件
    /// <para>在程序退出前调用，确保日志完整性</para>
    /// </summary>
    [DllImport(DLL_NAME, CallingConvention = CC)]
    public static extern void AOI_Logger_Flush();
}