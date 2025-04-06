import cv2
import os
import glob
import numpy as np
import traceback
from pathlib import Path
from PIL import Image

def create_video_from_images(image_folder, output_path, fps=15, sort_numerically=True):
    """
    将指定文件夹中的所有图片合成为视频
    
    Args:
        image_folder: 包含图片的文件夹路径
        output_path: 输出视频的路径
        fps: 帧率，默认为15
        sort_numerically: 是否按照文件名中的数字排序
    """
    try:
        print(f"正在处理图片文件夹: {image_folder}")
        
        # 获取所有图片文件
        image_files = glob.glob(os.path.join(image_folder, '*.jpg')) + \
                     glob.glob(os.path.join(image_folder, '*.jpeg')) + \
                     glob.glob(os.path.join(image_folder, '*.png'))
        
        if len(image_files) == 0:
            print(f"错误: 在 {image_folder} 中没有找到图片文件")
            return False
        
        print(f"找到 {len(image_files)} 个图片文件")
        
        # 按文件名排序
        if sort_numerically:
            # 尝试按文件名中的数字进行排序
            try:
                image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
                print("已按文件名中的数字排序图片")
            except Exception as e:
                print(f"按数字排序失败, 使用字母顺序排序: {str(e)}")
                image_files.sort()
        else:
            image_files.sort()
            print("已按文件名字母顺序排序图片")
        
        # 读取第一张图片以获取尺寸
        first_image = image_files[0]
        print(f"读取第一张图片以获取尺寸: {first_image}")
        
        # 使用PIL库读取图片，避免中文路径问题
        try:
            pil_img = Image.open(first_image)
            width, height = pil_img.size
            print(f"图片尺寸: {width}x{height}")
        except Exception as e:
            print(f"使用PIL读取图片失败: {str(e)}")
            print("尝试使用OpenCV读取...")
            img = cv2.imread(first_image)
            if img is None:
                print(f"错误: 无法读取图片 {first_image}")
                return False
            height, width, _ = img.shape
            print(f"图片尺寸: {width}x{height}")
        
        size = (width, height)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 也可以使用 'mp4v' 或其他编码格式
        print(f"创建视频写入器, 输出文件: {output_path}")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        if not video_writer.isOpened():
            print(f"错误: 无法创建视频文件 {output_path}")
            return False
        
        # 遍历所有图片并写入视频
        success_count = 0
        for i, image_file in enumerate(image_files):
            if i % 10 == 0:  # 只打印每10张图片的进度
                print(f"处理图片 {i+1}/{len(image_files)}")
            
            try:
                # 使用PIL读取图片，然后转换为OpenCV格式
                pil_img = Image.open(image_file)
                # 确保尺寸一致
                if pil_img.size[0] != width or pil_img.size[1] != height:
                    pil_img = pil_img.resize((width, height))
                
                # 将PIL图像转换为OpenCV格式
                img = np.array(pil_img)
                # 如果是RGB图像，将其转换为BGR (OpenCV使用BGR)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                video_writer.write(img)
                success_count += 1
            except Exception as e:
                print(f"处理图片 {image_file} 时出错: {str(e)}")
        
        # 释放资源
        video_writer.release()
        print(f"处理完成! 成功处理 {success_count}/{len(image_files)} 张图片")
        print(f"视频已成功保存到: {output_path}")
        return True
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        # 定义图片文件夹和输出视频路径
        # 获取当前脚本的目录
        current_dir = Path(__file__).parent
        
        # 图片文件夹路径
        image_folder = os.path.join(current_dir, "weed")
        
        # 输出视频路径 - 使用mp4格式可能在更多设备上兼容
        output_video = os.path.join(current_dir, "weed_video.mp4")
        
        print("开始将图片合成为视频...")
        print(f"图片文件夹: {image_folder}")
        print(f"输出视频: {output_video}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        
        # 创建视频
        result = create_video_from_images(image_folder, output_video, fps=10)
        
        if result:
            print("视频创建成功!")
        else:
            print("视频创建失败!")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        print(traceback.format_exc()) 