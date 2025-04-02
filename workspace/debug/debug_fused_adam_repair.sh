A=/usr/local/lib/python3.11/dist-packages/tensorflow/include/third_party/gpus/cuda/include
B=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangyikun-240108120104/software/miniconda3/envs/open-rlhf/lib/python3.11/site-packages/deepspeed/ops/csrc/adam

find ${A} -maxdepth 1 -type f ! -exec test -e ${B}/{} \; -exec ln -s {} ${B} \;