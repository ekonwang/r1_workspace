"""
Convert a tensorboard log to a dictionary
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Dict, List, Union
import os
import sys
def read_tensorboard_data(tb_path: str) -> Dict[str, List[Union[float, int]]]:
    """
    读取 TensorBoard 事件文件中的数据并返回字典格式
    
    参数:
        tb_path (str): TensorBoard 事件文件的路径（可以是目录或具体文件）
    
    返回:
        Dict[str, List[Union[float, int]]]: 包含所有标量数据的字典，
            键是标签名，值是对应的值列表
    """
    # 如果路径是目录，找到目录中的第一个事件文件
    if os.path.isdir(tb_path):
        event_files = [f for f in os.listdir(tb_path) if f.startswith('events.out.tfevents')]
        if not event_files:
            raise ValueError(f"在目录 {tb_path} 中找不到 TensorBoard 事件文件")
        tb_path = os.path.join(tb_path, event_files[0])
    
    # 初始化事件累积器
    event_acc = EventAccumulator(tb_path)
    event_acc.Reload()  # 加载所有事件数据
    
    # 获取所有标量标签
    tags = event_acc.Tags()['scalars']
    
    # 构建结果字典
    result = {}
    for tag in tags:
        # 获取每个标签的所有事件
        events = event_acc.Scalars(tag)
        # 提取值并存入列表
        result[tag] = [event.value for event in events]
        # 如果需要步数或时间戳，可以这样添加:
        # result[f"{tag}_steps"] = [event.step for event in events]
        # result[f"{tag}_wall_time"] = [event.wall_time for event in events]
    
    return result


# 示例用法
if __name__ == "__main__":
    tb_data = read_tensorboard_data(sys.argv[1])
    for tag, values in tb_data.items():
        print(f"{tag}: {values[:5]}... (共 {len(values)} 个数据点)")

