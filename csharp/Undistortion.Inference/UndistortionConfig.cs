namespace Undistortion.Inference;

/// <summary>
/// 去畸变初始化参数 —— <b>值类型结构体</b>，是所有初始化参数的唯一载体，也是唯一的“配置”概念。
///
/// <para><b>字段总表：必填 / GUI 暴露 / 默认值</b></para>
/// <list type="table">
///   <listheader>
///     <term>字段</term>
///     <description>必填 | GUI | 默认值 | 一句话</description>
///   </listheader>
///   <item>
///     <term>MatrixDir</term>
///     <description><b>必填</b> | <b>必须开放</b>（目录选择） | 无 | 12 个 .bin 的标定矩阵目录</description>
///   </item>
///   <item>
///     <term>Width</term>
///     <description><b>必填</b> | 只读展示（随图像/相机带出） | 无 | 目标图宽</description>
///   </item>
///   <item>
///     <term>Height</term>
///     <description><b>必填</b> | 只读展示（随图像/相机带出） | 无 | 目标图高</description>
///   </item>
///   <item>
///     <term>DeviceId</term>
///     <description>可选 | <b>必须开放</b>（多卡选卡） | <c>0</c> | CUDA 设备号</description>
///   </item>
/// </list>
///
/// <para><b>不在这里的东西（刻意不暴露，避免误配）</b></para>
/// <list type="bullet">
///   <item>日志级别：交付层固定 <c>log_level = 0</c>（日志只服务离线调用）；</item>
///   <item>引擎缓存：交付层固定 <c>null</c>（每次现场构建，≈6.7 s）；</item>
///   <item>通道数：固定 1（只支持单通道 <c>CV_8UC1</c>）；</item>
///   <item>搬运模式：固定“库内页锁暂存”，不足时自动回退 pageable。</item>
/// </list>
///
/// <para><b>构造方式</b>：<c>readonly struct</c> + <c>init</c> 访问器，只在对象初始化器里赋值，
/// 构造完成后不可改（要换尺寸/换矩阵就重建实例）。</para>
///
/// <code>
/// var config = new UndistortionConfig
/// {
///     MatrixDir = @"D:\opt\undistortion\botong25",   // 必填，GUI 目录选择
///     Width     = src.Width,                         // 必填，来自图像/相机
///     Height    = src.Height,
///     DeviceId  = 0,                                 // 可选，GUI 选卡
/// };
/// using var undistorter = new Undistorter(config);
/// </code>
/// </summary>
public readonly struct UndistortionConfig
{
    /// <summary>
    /// 上位机 GUI <b>必须提供可编辑控件</b>的字段名：<see cref="MatrixDir"/>（目录选择）、
    /// <see cref="DeviceId"/>（多卡选卡）。
    ///
    /// <para>给上位机生成配置界面时按这个列表建控件，不要手写字段名 —— 字段改名/增删时这里会同步暴露编译错误。</para>
    /// </summary>
    public static readonly string[] GuiEditableFields = new[] { nameof(MatrixDir), nameof(DeviceId) };

    /// <summary>
    /// 上位机 GUI <b>只读展示</b>的字段名：<see cref="Width"/>、<see cref="Height"/>。
    ///
    /// <para>尺寸由相机/图像分辨率带出，展示给操作员核对即可；允许编辑只会制造“尺寸与矩阵不匹配”的错误。</para>
    /// </summary>
    public static readonly string[] GuiReadOnlyFields = new[] { nameof(Width), nameof(Height) };

    /// <summary>
    /// 去畸变矩阵目录（UTF-8 路径）。<b>必填</b>；<b>GUI 必须开放</b>（目录选择框）。
    ///
    /// <para>目录内固定 12 个文件：<c>U_1..U_4</c> / <c>V_1..V_4</c> / <c>A1..A4</c>（各 width*height*4 字节）。</para>
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    ///   <item><b>必填</b>：为空 → <see cref="ArgumentException"/>；目录不存在 →
    ///         <see cref="DirectoryNotFoundException"/>。</item>
    ///   <item><b>GUI</b>：<b>必须开放</b>。不同机型/镜头对应不同矩阵，操作员必须能选；
    ///         选完立即校验目录里 12 个 <c>.bin</c> 是否齐全（齐全性最终由原生校验，
    ///         缺失返回 <see cref="UndistErrorCode.MatrixMissing"/>）。</item>
    ///   <item><b>取值</b>：优先绝对路径。相对路径由<b>调用方</b>按自己的基准解析
    ///         （本仓库的离线工具按 exe 目录解析）；库本身原样交给原生库。</item>
    ///   <item><b>不要给默认值</b>：猜错矩阵比报错更糟。</item>
    /// </list>
    /// </remarks>
    public string MatrixDir { get; init; }

    /// <summary>
    /// 目标图宽度（像素）。<b>必填</b>；<b>GUI 只读展示</b>（由相机/图像分辨率带出）。
    ///
    /// <para>必须等于矩阵宽度，也等于每帧 <c>Mat.Width</c>。</para>
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    ///   <item><b>必填</b>：<c>&lt;= 0</c> → <see cref="ArgumentOutOfRangeException"/>。</item>
    ///   <item><b>GUI</b>：<b>只读展示</b>即可 —— 尺寸来自相机配置/首帧图像，不该让操作员手改；
    ///         只有“同一套软件要支持多分辨率相机”时才做成可选（并在切换时重载实例）。
    ///         界面上建议显示“期望 vs 实际”，避免与矩阵不匹配。</item>
    ///   <item><b>校验</b>：与矩阵尺寸不一致 → 原生返回 <see cref="UndistErrorCode.MatrixSize"/>；
    ///         与每帧图像不一致 → <see cref="ArgumentException"/>。</item>
    /// </list>
    /// </remarks>
    public int Width { get; init; }

    /// <summary>
    /// 目标图高度（像素）。<b>必填</b>；<b>GUI 只读展示</b>（由相机/图像分辨率带出）。
    ///
    /// <para>必须等于矩阵高度，也等于每帧 <c>Mat.Height</c>。</para>
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    ///   <item><b>必填</b>：<c>&lt;= 0</c> → <see cref="ArgumentOutOfRangeException"/>。</item>
    ///   <item><b>GUI</b>：同 <see cref="Width"/>，只读展示。</item>
    ///   <item><b>校验</b>：同 <see cref="Width"/>。</item>
    /// </list>
    /// </remarks>
    public int Height { get; init; }

    /// <summary>
    /// CUDA 设备号（从 0 开始）。可选，默认 <c>0</c>；<b>GUI 必须开放</b>（多卡机器选卡）。
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    ///   <item><b>必填</b>：否。默认 <c>0</c>（单卡机器不用填）。</item>
    ///   <item><b>GUI</b>：<b>必须开放</b>（下拉框），否则多卡机器只能改代码/改配置文件；
    ///         下拉项建议用“序号 + 型号”（如 <c>0 - RTX 5060</c>），由上位机自行枚举。</item>
    ///   <item><b>校验</b>：负数 → <see cref="ArgumentOutOfRangeException"/>；
    ///         越界（&gt; 卡数）由原生的 CUDA 初始化失败兜底，
    ///         返回 <see cref="UndistErrorCode.InitFailed"/>。</item>
    ///   <item><b>注意</b>：每张卡配一个 <see cref="Undistorter"/> 实例；显存约 280 MB/实例。</item>
    /// </list>
    /// </remarks>
    public int DeviceId { get; init; }

    /// <summary>
    /// 字段级校验（不碰 GPU）。<see cref="Undistorter"/> 构造时先调用它，
    /// 让“参数写错”和“引擎起不来”得到两种不同的错误。
    /// </summary>
    /// <exception cref="ArgumentException"><see cref="MatrixDir"/> 为空</exception>
    /// <exception cref="DirectoryNotFoundException"><see cref="MatrixDir"/> 指向的目录不存在</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    ///     <see cref="Width"/> / <see cref="Height"/> 非正，或 <see cref="DeviceId"/> 为负
    /// </exception>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(MatrixDir))
            throw new ArgumentException("MatrixDir 不能为空", nameof(MatrixDir));
        if (!Directory.Exists(MatrixDir))
            throw new DirectoryNotFoundException($"矩阵目录不存在: {MatrixDir}");
        if (Width <= 0 || Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(Width), $"宽高必须为正，实际 {Width}x{Height}");
        if (DeviceId < 0)
            throw new ArgumentOutOfRangeException(nameof(DeviceId), $"设备号不能为负: {DeviceId}");
    }

    public override string ToString() =>
        $"{Width}x{Height}, matrix={MatrixDir}, device={DeviceId}";
}
