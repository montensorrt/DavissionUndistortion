#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像差异比较工具
比较两个目录下同名图像的像素差异，统计差异信息并输出报告
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# 尝试导入图像处理库
try:
    import cv2
except ImportError:
    print("错误: 请安装 opencv-python: pip install opencv-python")
    sys.exit(1)

# 支持的图像格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def find_images(directory: str) -> dict:
    """递归查找目录下所有图像文件，返回 {文件名: 完整路径} 字典"""
    images = {}
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                # 使用相对路径作为键，保持目录结构
                rel_path = os.path.relpath(os.path.join(root, f), directory)
                images[rel_path] = os.path.join(root, f)
    return images


def compare_images(img1_path: str, img2_path: str) -> tuple:
    """比较两张图像的像素差异，返回(统计信息, 差异图)"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
    
    if img1 is None:
        return {'error': f'无法读取图像: {img1_path}'}, None
    if img2 is None:
        return {'error': f'无法读取图像: {img2_path}'}, None
    
    # 检查尺寸一致性
    if img1.shape != img2.shape:
        return {'error': f'尺寸不匹配: {img1.shape} vs {img2.shape}'}, None
    
    # 计算像素差异 (使用int16避免溢出，取绝对值防止-1变为254)
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    
    # 生成差异图 - 使用膨胀操作放大差异点，使稀疏差异可见
    # 规则: >100 → 255(白), >50 → 200, >10 → 150, >0 → 100, =0 → 0(黑)
    diff_image = np.zeros_like(diff, dtype=np.uint8)
    diff_image[diff > 0] = 100     # 差异>0: 灰色
    diff_image[diff > 10] = 150    # 差异>10: 较亮
    diff_image[diff > 50] = 200    # 差异>50: 亮
    diff_image[diff > 100] = 255   # 差异>100: 白色
    
    # 使用膨胀操作放大差异点（因为差异像素非常稀疏，单个像素难以看见）
    # 膨胀核大小: 5x5，使每个差异点扩展为5x5的区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    diff_image = cv2.dilate(diff_image, kernel, iterations=1)
    
    # 统计信息
    total_pixels = diff.size
    diff_mask = diff > 0
    diff_count = np.sum(diff_mask)
    diff_gt2_count = np.sum(diff > 2)  # 差异>2的像素数量
    
    # 计算差异像素的平均差异值（仅统计有差异的像素）
    mean_diff_only = float(np.mean(diff[diff_mask])) if diff_count > 0 else 0.0
    
    return {
        'shape': img1.shape,
        'total_pixels': total_pixels,
        'diff_count': int(diff_count),
        'diff_ratio': diff_count / total_pixels * 100,
        'diff_gt2_count': int(diff_gt2_count),  # 差异>2的像素数量
        'diff_gt2_ratio': diff_gt2_count / total_pixels * 100,
        'max_diff': int(np.max(diff)),
        'min_diff': int(np.min(diff[diff_mask])) if diff_count > 0 else 0,
        'mean_diff_all': float(np.mean(diff)),  # 全部像素的平均差异
        'mean_diff_only': mean_diff_only,  # 仅差异像素的平均差异
        'std_diff': float(np.std(diff)),
        'identical': diff_count == 0
    }, diff_image


def format_result(name: str, stats: dict) -> str:
    """格式化单个比较结果"""
    if 'error' in stats:
        return f"[{name}]\n  错误: {stats['error']}\n"
    
    lines = [
        f"[{name}]",
        f"  尺寸: {stats['shape']}",
        f"  总像素数: {stats['total_pixels']:,}",
        f"  差异像素数(>0): {stats['diff_count']:,} ({stats['diff_ratio']:.4f}%)",
        f"  差异像素数(>2): {stats['diff_gt2_count']:,} ({stats['diff_gt2_ratio']:.4f}%)",
        f"  最大差异: {stats['max_diff']}",
        f"  最小差异: {stats['min_diff']}",
        f"  平均差异(全部像素): {stats['mean_diff_all']:.4f}  ← 分母=总像素数",
        f"  平均差异(仅差异像素): {stats['mean_diff_only']:.4f}  ← 分母=差异像素数",
        f"  标准差: {stats['std_diff']:.4f}",
        f"  完全一致: {'是' if stats['identical'] else '否'}",
        ""
    ]
    return '\n'.join(lines)


def save_diff_image(diff_image: np.ndarray, name: str, diff_dir: str) -> str:
    """保存差异图到指定目录"""
    # 保持原目录结构
    output_path = os.path.join(diff_dir, name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 修改文件名，添加_diff后缀
    base, ext = os.path.splitext(output_path)
    output_path = f"{base}_diff{ext}"
    
    cv2.imwrite(output_path, diff_image)
    return output_path


def main(dir1: str, dir2: str, output_file: str = None, diff_dir: str = None):
    """主函数：比较两个目录下的同名图像"""
    print(f"=" * 60)
    print(f"图像差异比较工具")
    print(f"=" * 60)
    print(f"目录1: {dir1}")
    print(f"目录2: {dir2}")
    if diff_dir:
        print(f"差异图保存目录: {diff_dir}")
    print(f"-" * 60)
    
    # 创建差异图保存目录
    if diff_dir:
        os.makedirs(diff_dir, exist_ok=True)
    
    # 查找图像
    images1 = find_images(dir1)
    images2 = find_images(dir2)
    
    print(f"目录1 图像数量: {len(images1)}")
    print(f"目录2 图像数量: {len(images2)}")
    
    # 找出同名文件
    common_names = set(images1.keys()) & set(images2.keys())
    only_in_dir1 = set(images1.keys()) - set(images2.keys())
    only_in_dir2 = set(images2.keys()) - set(images1.keys())
    
    print(f"同名图像数量: {len(common_names)}")
    print(f"-" * 60)
    
    # 比较结果
    results = []
    summary = {'total': len(common_names), 'identical': 0, 'different': 0, 'error': 0}
    saved_diff_count = 0
    
    for name in sorted(common_names):
        stats, diff_image = compare_images(images1[name], images2[name])
        result_str = format_result(name, stats)
        results.append(result_str)
        print(result_str)
        
        if 'error' in stats:
            summary['error'] += 1
        elif stats['identical']:
            summary['identical'] += 1
        else:
            summary['different'] += 1
            # 保存差异图（仅保存有差异的图像）
            if diff_dir and diff_image is not None:
                saved_path = save_diff_image(diff_image, name, diff_dir)
                print(f"  → 差异图已保存: {saved_path}")
                saved_diff_count += 1
    
    # 汇总信息
    summary_lines = [
        "=" * 60,
        "汇总统计",
        "=" * 60,
        f"比较图像总数: {summary['total']}",
        f"完全一致: {summary['identical']}",
        f"存在差异: {summary['different']}",
        f"处理错误: {summary['error']}",
    ]
    
    if diff_dir:
        summary_lines.append(f"已保存差异图: {saved_diff_count} 张")
    
    if only_in_dir1:
        summary_lines.append(f"\n仅存在于目录1 ({len(only_in_dir1)} 个):")
        for name in sorted(only_in_dir1)[:10]:
            summary_lines.append(f"  - {name}")
        if len(only_in_dir1) > 10:
            summary_lines.append(f"  ... 及其他 {len(only_in_dir1) - 10} 个文件")
    
    if only_in_dir2:
        summary_lines.append(f"\n仅存在于目录2 ({len(only_in_dir2)} 个):")
        for name in sorted(only_in_dir2)[:10]:
            summary_lines.append(f"  - {name}")
        if len(only_in_dir2) > 10:
            summary_lines.append(f"  ... 及其他 {len(only_in_dir2) - 10} 个文件")
    
    summary_str = '\n'.join(summary_lines)
    print(summary_str)
    
    # 保存结果
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"compare_result_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"图像差异比较报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目录1: {dir1}\n")
        f.write(f"目录2: {dir2}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write('\n'.join(results))
        f.write(f"\n{summary_str}\n")
    
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    # 默认路径配置（可根据需要修改）
    DEFAULT_DIR1 = r"E:\algrithm\project\2DAOI\feipai_input\product_2\1022\143323\undis_raw\demo"
    DEFAULT_DIR2 = r"E:\algrithm\project\2DAOI\feipai_input\product_2\1022\143323\undistorted\C1R1C1"
    DEFAULT_DIFF_DIR = r"E:\algrithm\project\2DAOI\feipai_input\product_2\1022\143323\diff_mask"
    
    if len(sys.argv) >= 3:
        dir1, dir2 = sys.argv[1], sys.argv[2]
        output = sys.argv[3] if len(sys.argv) > 3 else None
        diff_dir = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_DIFF_DIR
    else:
        print("使用方法: python compare_images.py <目录1> <目录2> [输出文件] [差异图目录]")
        print(f"未提供参数，使用默认路径...")
        dir1, dir2 = DEFAULT_DIR1, DEFAULT_DIR2
        output = None
        diff_dir = DEFAULT_DIFF_DIR
    
    if not os.path.isdir(dir1):
        print(f"错误: 目录不存在 - {dir1}")
        sys.exit(1)
    if not os.path.isdir(dir2):
        print(f"错误: 目录不存在 - {dir2}")
        sys.exit(1)
    
    main(dir1, dir2, output, diff_dir)
