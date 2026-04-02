import torch
import ultralytics
print(f'Ultralytics 版本：{ultralytics.__version__}')
print(f'Torch 版本：{torch.__version__}')
print(f'CUDA 可用：{torch.cuda.is_available()}')
print('环境检查通过！')
