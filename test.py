import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import os


def generate_industrial_data():
    print("开始生成工业数据...")

    # 读取CSV文件
    df = pd.read_csv('位号2.csv')

    # 创建输出目录
    os.makedirs('output', exist_ok=True)

    # 时间范围配置
    start_time = datetime(2025, 12, 28, 0, 0, 0)
    end_time = datetime(2026, 1, 17, 23, 59, 59)

    # 计算总分钟数
    total_minutes = int((end_time - start_time).total_seconds() / 60) + 1
    print(f"时间范围: {start_time} 到 {end_time}")
    print(f"总分钟数: {total_minutes}")
    print(f"总点位数量: {len(df)}")
    print(f"预计总数据量: {len(df) * total_minutes:,} 条")

    # 生成所有时间点
    timestamps = [start_time + timedelta(minutes=i) for i in range(total_minutes)]

    # 转换为纳米时间戳（Unix timestamp in nanoseconds）
    nano_timestamps = [int(ts.timestamp() * 1e9) for ts in timestamps]

    # 分批处理，避免内存问题
    batch_size = 100  # 每次处理100个点位
    total_points = len(df)

    # 准备输出文件
    output_file = 'output/industrial_data.json'

    # 如果文件已存在，删除它
    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"开始生成数据，输出到: {output_file}")
    start_processing_time = time.time()

    # 打开文件准备写入
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')  # 开始JSON数组

        # 分批处理点位
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_df = df.iloc[batch_start:batch_end]

            print(f"处理点位 {batch_start + 1} 到 {batch_end}...")
            batch_start_time = time.time()

            batch_data = []

            # 为每个点位生成数据
            for idx, row in batch_df.iterrows():
                tag_node = f"{row['group_name']}.{row['name']}"

                # 为每个时间点生成数据
                for i, nano_ts in enumerate(nano_timestamps):
                    # 为每个时间点生成0-1000的随机值
                    random_value = np.random.uniform(0, 1000)

                    # 构建数据记录
                    record = {
                        "tag_node": tag_node,
                        "measurement_name": "20260101",
                        "value": float(random_value),
                        "time": nano_ts
                    }

                    batch_data.append(record)

                    # 每10000条记录写入一次，避免内存占用过大
                    if len(batch_data) >= 10000:
                        # 写入到文件
                        for j, record in enumerate(batch_data):
                            json_str = json.dumps(record, ensure_ascii=False)
                            if not (batch_start == 0 and i == 0 and j == 0):
                                f.write(',\n')
                            f.write('  ' + json_str)
                        batch_data = []

                # 显示进度
                if (idx - batch_start) % 10 == 0:
                    print(f"  已处理点位 {idx + 1}/{total_points}")

            # 写入剩余的数据
            if batch_data:
                for j, record in enumerate(batch_data):
                    json_str = json.dumps(record, ensure_ascii=False)
                    # 检查是否需要添加逗号
                    if not (batch_start == 0 and j == 0):
                        f.write(',\n')
                    f.write('  ' + json_str)

            batch_time = time.time() - batch_start_time
            print(f"  批次处理完成，用时: {batch_time:.2f}秒")

        f.write('\n]')  # 结束JSON数组

    total_processing_time = time.time() - start_processing_time
    print(f"\n数据生成完成!")
    print(f"总处理时间: {total_processing_time:.2f}秒")

    # 计算文件大小
    file_size = os.path.getsize(output_file)
    print(f"生成的文件大小: {file_size:,} 字节 ({file_size / 1024 / 1024:.2f} MB)")

    # 验证数据
    print("\n验证数据...")
    with open(output_file, 'r', encoding='utf-8') as f:
        # 读取前几行验证
        lines = []
        for i in range(5):
            lines.append(f.readline().strip())

        # 检查文件结构
        print("文件前几行:")
        for line in lines:
            if line:  # 跳过空行
                print(f"  {line[:100]}...")

    return output_file


def generate_sample_data_for_verification():
    """生成一个小样本用于验证数据结构"""
    print("\n生成样本数据用于验证...")

    # 读取CSV文件
    df = pd.read_csv('位号2.csv')

    # 只取前3个点位，前3个时间点
    sample_points = df.head(3)
    sample_times = [
        datetime(2025, 12, 26, 0, 0, 0),
        datetime(2025, 12, 26, 0, 1, 0),
        datetime(2025, 12, 26, 0, 2, 0)
    ]

    sample_data = []

    for idx, row in sample_points.iterrows():
        tag_node = f"{row['group_name']}.{row['name']}"

        for sample_time in sample_times:
            nano_ts = int(sample_time.timestamp() * 1e9)
            random_value = np.random.uniform(0, 1000)

            record = {
                "tag_node": tag_node,
                "measurement_name": "20260101",
                "value": float(random_value),
                "time": nano_ts
            }
            sample_data.append(record)

    # 保存样本数据
    sample_file = 'output/sample_data.json'
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"样本数据已保存到: {sample_file}")
    print("样本数据示例:")
    for i, record in enumerate(sample_data[:3]):
        print(f"  记录 {i + 1}: {json.dumps(record, ensure_ascii=False)}")

    return sample_file


def main():
    print("=" * 60)
    print("工业数据生成器")
    print("=" * 60)

    # 检查输入文件
    if not os.path.exists('位号2.csv'):
        print("错误: 未找到 '点位号.csv' 文件!")
        print("请确保 '点位号.csv' 文件在当前目录下")
        return

    try:
        # 先生成一个样本用于验证
        generate_sample_data_for_verification()

        # 询问是否生成完整数据
        print("\n" + "=" * 60)
        user_input = input("是否要生成完整数据（8.6百万条，文件会很大）? (y/n): ")

        if user_input.lower() == 'y':
            # 生成完整数据
            output_file = generate_industrial_data()

            print("\n" + "=" * 60)
            print("操作完成!")
            print(f"1. 完整数据: {output_file}")
            print(f"2. 样本数据: output/sample_data.json")
            print("\n注意事项:")
            print("- 完整JSON文件非常大，处理时请确保有足够磁盘空间")
            print("- 建议使用专业工具或数据库加载和处理这些数据")
            print("- 可以使用 gzip 压缩文件以节省空间")
        else:
            print("已取消生成完整数据。")
            print("样本数据已生成在 output/sample_data.json")

    except Exception as e:
        print(f"生成数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
