namespace Undistortion.Inference;

/// <summary>运行期信息（C-API <c>undist_info</c> 的托管映射，用于自检与日志）</summary>
public sealed class UndistortionInfo
{
    /// <summary>图宽（= 矩阵宽）</summary>
    public int Width { get; init; }

    /// <summary>图高（= 矩阵高）</summary>
    public int Height { get; init; }

    /// <summary>通道数（固定 1）</summary>
    public int Channels { get; init; }

    /// <summary>常驻显存的表字节数（12 B/px）</summary>
    public long TableBytes { get; init; }

    /// <summary>序列化引擎字节数</summary>
    public long EngineBytes { get; init; }

    /// <summary>显存总占用（表 + 输入/输出缓冲）</summary>
    public long DeviceBytes { get; init; }

    /// <summary>每像素表开销，应为 12.0</summary>
    public double TableBytesPerPixel { get; init; }

    /// <summary>矩阵加载 + 紧凑表构建耗时（ms）</summary>
    public double InitTableMs { get; init; }

    /// <summary>引擎构建/加载 + 表上显存 + 主机暂存耗时（ms）</summary>
    public double InitEngineMs { get; init; }

    /// <summary>初始化总耗时（ms）</summary>
    public double InitTotalMs { get; init; }

    /// <summary>生效的主机搬运方式：true = 库内页锁暂存，false = 页锁内存不足已回退 pageable</summary>
    public bool PinnedStaging { get; init; }

    /// <summary>GPU SM 数</summary>
    public int SmCount { get; init; }

    /// <summary>GPU 名称</summary>
    public string GpuName { get; init; } = string.Empty;

    /// <summary>TensorRT 版本</summary>
    public string TensorRtVersion { get; init; } = string.Empty;

    public override string ToString() =>
        $"{Width}x{Height} x{Channels}, table={TableBytes / 1048576.0:F1} MB " +
        $"({TableBytesPerPixel:F1} B/px), device={DeviceBytes / 1048576.0:F1} MB, " +
        $"staging={(PinnedStaging ? "pinned" : "pageable")}, " +
        $"gpu={GpuName} ({SmCount} SM), trt={TensorRtVersion}";
}

/// <summary>最近一次推理的分段耗时（毫秒；串行口径）</summary>
public sealed class UndistortionTiming
{
    /// <summary>主机 → 设备</summary>
    public double H2dMs { get; init; }

    /// <summary>enqueueV3 + 流同步（纯 kernel）</summary>
    public double InferMs { get; init; }

    /// <summary>设备 → 主机</summary>
    public double D2hMs { get; init; }

    /// <summary>整个调用（含库内主机行拷贝）</summary>
    public double TotalMs { get; init; }

    public override string ToString() =>
        $"h2d={H2dMs:F3} infer={InferMs:F3} d2h={D2hMs:F3} total={TotalMs:F3} ms";
}
