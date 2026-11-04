using OpenCvSharp;

namespace Undistortion.Tools;

/// <summary>灰度图逐像素比对结果（与 C++ 验收口径一致）</summary>
public sealed class ImageDiff
{
    /// <summary>差异像素数（|a-b| &gt; 0）</summary>
    public long DiffPixels { get; init; }

    /// <summary>最大绝对差（LSB）</summary>
    public int MaxAbsDiff { get; init; }

    /// <summary>差异 ≥ 2 LSB 的像素数</summary>
    public long CountGe2 { get; init; }

    /// <summary>平均绝对误差</summary>
    public double Mae { get; init; }

    /// <summary>峰值信噪比（dB）</summary>
    public double PsnrDb { get; init; }
}

/// <summary>验收比对工具：只在工具项目中使用，不进交付库。</summary>
public static class ImageComparer
{
    /// <summary>逐像素比对两张同尺寸 <c>CV_8UC1</c> 图。</summary>
    /// <exception cref="ArgumentException">尺寸或通道不一致</exception>
    public static ImageDiff Compare(Mat a, Mat b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Type() != MatType.CV_8UC1 || b.Type() != MatType.CV_8UC1 || a.Size() != b.Size())
            throw new ArgumentException("Compare 只接受同尺寸 CV_8UC1 图像");

        using var diff = new Mat();
        Cv2.Absdiff(a, b, diff);

        long nonzero = Cv2.CountNonZero(diff);
        Cv2.MinMaxLoc(diff, out _, out double max);
        double mae = Cv2.Mean(diff).Val0;

        using var mask = new Mat();
        Cv2.Threshold(diff, mask, 1, 255, ThresholdTypes.Binary);
        long ge2 = Cv2.CountNonZero(mask);

        using var diff32 = new Mat();
        diff.ConvertTo(diff32, MatType.CV_32F);
        using var squared = new Mat();
        Cv2.Multiply(diff32, diff32, squared);
        double mse = Cv2.Mean(squared).Val0;
        double psnr = mse <= 0 ? 999.0 : 10.0 * Math.Log10(255.0 * 255.0 / mse);

        return new ImageDiff
        {
            DiffPixels = nonzero,
            MaxAbsDiff = (int)max,
            CountGe2 = ge2,
            Mae = mae,
            PsnrDb = psnr,
        };
    }
}
