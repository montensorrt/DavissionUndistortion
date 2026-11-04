using OpenCvSharp;

namespace Undistortion.Inference;

/// <summary>
/// 去畸变（TensorRT）推理节点 —— Mat 进、Mat 出，全程内存态，不落盘。
///
/// 初始化参数是值类型结构体 <see cref="UndistortionConfig"/>，由调用方构造：
/// <code>
/// var config = new UndistortionConfig { MatrixDir = @"D:\opt\botong25", Width = 5120, Height = 4096 };
/// using var undistorter = new Undistorter(config);
///
/// using Mat src = Cv2.ImRead("raw.bmp", ImreadModes.Grayscale);
/// using Mat dst = undistorter.Process(src);        // 或 Process(src, 复用缓冲)
/// Cv2.ImWrite("undis.bmp", dst);
/// </code>
///
/// 约定：
///   <list type="bullet">
///     <item>只支持单通道 <c>CV_8UC1</c>，尺寸必须等于配置中的 <c>Width × Height</c>；</item>
///     <item>单实例单线程：原生侧一条 stream + 互斥锁，多线程不会崩但会串行；</item>
///     <item>忘记 Dispose 时由终结器兜底释放显存（仅兜底，请优先 using）。</item>
///   </list>
///
/// 主机搬运固定为「库内页锁暂存」（12 B/px 表 + 2 × W×H 暂存），页锁内存不足时原生侧
/// 自动回退 pageable，可通过 <see cref="UndistortionInfo.PinnedStaging"/> 观察。
/// 其余搬运方式（byte[] / 零拷贝 PinnedBuffer / 裸指针 / 纯 kernel 基准）已从交付层移除，
/// 口径与实测数据见 <c>document/UNDIST-105</c>。
///
/// <b>日志固定关闭</b>（原生 <c>log_level = 0</c>）：日志写 stderr，属开发/验收期诊断手段，
/// C# 上位机不启用；离线调用方（C++ 验收程序 / Python 批处理）在 C ABI 侧自行开启。
/// 排障请用 <see cref="Info"/>、<see cref="LastTiming"/> 与异常里的错误码/原生错误串。
/// </summary>
public sealed class Undistorter : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    /// <summary>运行期信息（尺寸 / 表与显存占用 / GPU / TRT 版本）</summary>
    public UndistortionInfo Info { get; }

    /// <summary>最近一次推理的分段耗时（纯原生口径，不含托管侧开销）</summary>
    public UndistortionTiming LastTiming { get; private set; } = new();

    /// <summary>
    /// 由配置初始化：加载 12 个矩阵 → 构建紧凑表 → 构建 TRT 引擎 → 表与缓冲上显存。
    /// </summary>
    /// <param name="config">初始化参数结构体（构造后不可变，字段级校验在此完成）</param>
    /// <exception cref="ArgumentException">配置字段非法（目录为空 / 宽高非正 / 设备号为负）</exception>
    /// <exception cref="DirectoryNotFoundException">矩阵目录不存在</exception>
    /// <exception cref="UndistortionException">原生初始化失败（矩阵缺失、尺寸不符、显存不足等）</exception>
    public Undistorter(UndistortionConfig config)
    {
        config.Validate();

        var options = new NativeMethods.OptionsNative
        {
            memory_mode = 1,            // 库内页锁暂存（交付层唯一口径）
            device_id = config.DeviceId,
            channels = 1,               // 仅支持单通道
            log_level = 0,              // 固定关闭：日志只服务离线调用方，上位机不启用
        };

        // 第 5 个参数是引擎缓存文件路径：交付层固定传 null（不缓存、每次现场构建 ≈6.7 s）。
        // 上位机「启动一次、长期运行」，这笔一次性成本可接受；缓存能力保留在 C ABI 侧，
        // 供离线 C++ 验收 / Python 批处理按需使用（document/UNDIST-105 §6.7）。
        int rc = NativeMethods.undist_create(config.MatrixDir, config.Width, config.Height,
                                             ref options, null, out _handle);
        if (rc != (int)UndistErrorCode.Ok)
        {
            // 失败路径原生也可能回写句柄：先取错误串再释放，避免泄漏表/显存
            string message = NativeMethods.ReadError(_handle);
            ReleaseHandle();
            throw new UndistortionException((UndistErrorCode)rc, message);
        }

        Info = QueryInfo();
    }

    /// <summary>处理一张图，返回新分配的输出 Mat（调用方负责 Dispose）。</summary>
    /// <param name="src">输入图，<c>CV_8UC1</c>，尺寸等于配置尺寸</param>
    public Mat Process(Mat src)
    {
        var dst = new Mat(Info.Height, Info.Width, MatType.CV_8UC1);
        try
        {
            Process(src, dst);
            return dst;
        }
        catch
        {
            dst.Dispose();
            throw;
        }
    }

    /// <summary>处理一张图，写入调用方复用的输出 Mat（高频路径用这个，避免每帧 20 MB 分配）。</summary>
    /// <param name="src">输入图，<c>CV_8UC1</c>，尺寸等于配置尺寸</param>
    /// <param name="dst">输出图，<c>CV_8UC1</c>，尺寸等于配置尺寸，必须连续</param>
    public void Process(Mat src, Mat dst)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(src);
        ArgumentNullException.ThrowIfNull(dst);

        // 非连续输入（ROI 视图等）临时克隆一份；连续输入零额外拷贝
        using Mat? clone = src.IsContinuous() ? null : src.Clone();
        Mat input = clone ?? src;

        ValidateImage(input, nameof(src));
        ValidateImage(dst, nameof(dst));

        int rc = NativeMethods.undist_process(_handle, input.Data, (int)input.Step(),
                                              dst.Data, (int)dst.Step(), 0,
                                              out NativeMethods.TimingNative timing);
        if (rc != (int)UndistErrorCode.Ok)
            throw new UndistortionException((UndistErrorCode)rc, NativeMethods.ReadError(_handle));

        LastTiming = new UndistortionTiming
        {
            H2dMs = timing.h2d_ms,
            InferMs = timing.infer_ms,
            D2hMs = timing.d2h_ms,
            TotalMs = timing.total_ms,
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        ReleaseHandle();
        GC.SuppressFinalize(this);
    }

    ~Undistorter() => ReleaseHandle();

    private UndistortionInfo QueryInfo()
    {
        int rc = NativeMethods.undist_get_info(_handle, out NativeMethods.InfoNative native);
        if (rc != (int)UndistErrorCode.Ok)
            throw new UndistortionException((UndistErrorCode)rc, NativeMethods.ReadError(_handle));

        string gpu;
        string trt;
        unsafe
        {
            gpu = NativeMethods.Utf8From(native.gpu_name, 96);
            trt = NativeMethods.Utf8From(native.tensorrt_version, 32);
        }

        return new UndistortionInfo
        {
            Width = native.width,
            Height = native.height,
            Channels = native.channels,
            TableBytes = native.table_bytes,
            EngineBytes = native.engine_bytes,
            DeviceBytes = native.device_bytes,
            TableBytesPerPixel = native.table_bytes_per_pixel,
            InitTableMs = native.init_table_ms,
            InitEngineMs = native.init_engine_ms,
            InitTotalMs = native.init_total_ms,
            PinnedStaging = native.memory_mode == 1,
            SmCount = native.sm_count,
            GpuName = gpu,
            TensorRtVersion = trt,
        };
    }

    private void ValidateImage(Mat mat, string paramName)
    {
        if (mat.Empty())
            throw new ArgumentException("图像为空", paramName);
        if (mat.Type() != MatType.CV_8UC1)
            throw new ArgumentException($"仅支持 CV_8UC1，实际 {mat.Type()}", paramName);
        if (mat.Width != Info.Width || mat.Height != Info.Height)
            throw new ArgumentException(
                $"尺寸不符：期望 {Info.Width}x{Info.Height}，实际 {mat.Width}x{mat.Height}", paramName);
        if (!mat.IsContinuous())
            throw new ArgumentException("必须是连续内存", paramName);
    }

    private void ReleaseHandle()
    {
        if (_handle == IntPtr.Zero) return;
        NativeMethods.undist_release(_handle);
        _handle = IntPtr.Zero;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(Undistorter));
    }
}
