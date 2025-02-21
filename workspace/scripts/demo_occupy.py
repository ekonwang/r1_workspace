import torch 
import time
import os
import argparse
import shutil
import sys
import numpy as np
import subprocess

from utils import print_error


def check_gpu_memory_already_occupied(threshold=32):
    assert threshold > 0

    try:
        # 使用nvidia-smi命令获取显存使用情况
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        # 获取显存使用量（单位为MiB）
        memory_array = result.stdout.strip().split('\n')
        memory_usage = float(np.mean(np.array([int(r) for r in memory_array])))
        # 判断是否超过32MB
        if memory_usage > threshold:
            print_error(f"暂缓进入 【OCCUPY】 模式，当前 {len(memory_array)}卡平均显存使用量: {memory_usage / 1024:.4f}GB.")
            return True 
        else:
            return False

    except subprocess.CalledProcessError:
        # 如果nvidia-smi命令执行失败，返回False
        print_error("无法获取显存使用情况。")
        return True

 
def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication')
    parser.add_argument('--gpus', help='gpu amount', required=True, type=int)
    parser.add_argument('--size', help='matrix size', default=70000, type=int)
    parser.add_argument('--interval', help='sleep interval', default=0.0001, type=float)
    parser.add_argument('--threshold', type=int, help='memory threshold for occupying', default=0)
    args = parser.parse_args()
    return args
 
 
def matrix_multiplication(args):
    print_error(f"进入 【OCCPY】 模式")
    while True:
        a_list, b_list, result = [], [], []    
        size = (args.size, args.size)

        for i in range(args.gpus):
            a_list.append(torch.rand(size, device=i))
            b_list.append(torch.rand(size, device=i))
            result.append(torch.rand(size, device=i))

        for i in range(args.gpus):
            result[i] = a_list[i] * b_list[i]
        time.sleep(args.interval)
 
 
if __name__ == "__main__":
    # usage: python demo_occupy.py --size 55000 --gpus 4 --interval 0.03 --threshold 24
    args = parse_args()

    while True:
        try:
            if args.threshold > 0:
                while(check_gpu_memory_already_occupied(args.threshold)):
                    time.sleep(5)
                matrix_multiplication(args)
            else:
                matrix_multiplication(args)
        except Exception as err:
            print_error(err)
            # Stay in loop until interrupt from the user.
            if 'keyboard' in str(err).lower():
                break
            time.sleep(5)
