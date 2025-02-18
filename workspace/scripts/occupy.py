import torch 
import time
import os
import argparse
import shutil
import sys
 
def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication')
    parser.add_argument('--gpus', help='gpu amount', required=True, type=int)
    parser.add_argument('--size', help='matrix size', default=70000, type=int)
    parser.add_argument('--interval', help='sleep interval', default=0.0001, type=float)
    args = parser.parse_args()
    return args
 
 
def matrix_multiplication(args):
    
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
    # usage: python occupy.py --size 55000 --gpus 4 --interval 0.03
    args = parse_args()
    matrix_multiplication(args)
