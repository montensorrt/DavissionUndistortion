using System.Runtime.InteropServices;

namespace Undistortion.Inference;

/// <summary>错误码（与 C-API <c>undist_error_code</c> 一一对应）</summary>
public enum UndistErrorCode
{
    Ok = 0,
    /// <summary>CUDA / TensorRT 初始化失败</summary>
    InitFailed = -1,
    /// <summary>句柄为空 / 已释放 / 未就绪</summary>
    InvalidHandle = -2,
    /// <summary>指针为空 / 宽高不符 / stride 非法</summary>
    InvalidImage = -3,
    /// <summary>enqueueV3 失败</summary>
    InferenceFailed = -4,
    /// <summary>表 / 引擎未就绪</summary>
    NotReady = -5,
    /// <summary>矩阵目录或 12 个 .bin 缺失</summary>
    MatrixMissing = -6,
    /// <summary>.bin 大小与 width*height*4 不符</summary>
    MatrixSize = -7,
    /// <summary>主机 / 显存分配失败</summary>
    OutOfMemory = -8,
    /// <summary>TRT 引擎构建或反序列化失败</summary>
    EngineBuild = -9,
    /// <summary>通道数不是 1 等未实现能力</summary>
    Unsupported = -10,
    /// <summary>出参为空 / 参数自相矛盾</summary>
    InvalidArg = -11,
}

/// <summary>
/// C ABI 的托管声明（internal，不对外暴露）。
///
/// 只声明交付层真正用到的 5 个导出；原生库中的基准测试 / 内存模式切换 / 页锁注册
/// 等能力不在此暴露，其口径见 <c>document/UNDIST-105</c>。
/// </summary>
internal static unsafe class NativeMethods
{
    private const string DllName = "undist_api";

    /// <summary>对应 C 侧 <c>undist_options</c>（4 × int32）</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct OptionsNative
    {
        public int memory_mode;
        public int device_id;
        public int channels;
        public int log_level;
    }

    /// <summary>对应 C 侧 <c>undist_info</c>（布局必须逐字节一致）</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct InfoNative
    {
        public long table_bytes;
        public long engine_bytes;
        public long device_bytes;
        public double table_bytes_per_pixel;
        public double init_table_ms;
        public double init_engine_ms;
        public double init_total_ms;
        public int width;
        public int height;
        public int channels;
        public int memory_mode;
        public int table_variant;
        public int sm_count;
        public fixed byte gpu_name[96];
        public fixed byte tensorrt_version[32];
    }

    /// <summary>对应 C 侧 <c>undist_timing</c>（4 × double）</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct TimingNative
    {
        public double h2d_ms;
        public double infer_ms;
        public double d2h_ms;
        public double total_ms;
    }

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int undist_create(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string matrixDir,
        int width, int height,
        ref OptionsNative options,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? engineCachePath,
        out IntPtr handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int undist_process(
        IntPtr handle, IntPtr src, int srcStride, IntPtr dst, int dstStride,
        int hostFlags, out TimingNative timing);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int undist_get_info(IntPtr handle, out InfoNative info);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr undist_get_error(IntPtr handle);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void undist_release(IntPtr handle);

    /// <summary>固定长度 UTF-8 字段 → 托管串</summary>
    internal static string Utf8From(byte* p, int capacity)
    {
        int len = 0;
        while (len < capacity && p[len] != 0) len++;
        return len == 0 ? string.Empty : System.Text.Encoding.UTF8.GetString(p, len);
    }

    /// <summary>取原生错误串（句柄为空时安全返回空串）</summary>
    internal static string ReadError(IntPtr handle)
    {
        IntPtr p = undist_get_error(handle);
        if (p == IntPtr.Zero) return string.Empty;

        int len = 0;
        while (Marshal.ReadByte(p, len) != 0) len++;
        if (len == 0) return string.Empty;

        var bytes = new byte[len];
        Marshal.Copy(p, bytes, 0, len);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }
}
