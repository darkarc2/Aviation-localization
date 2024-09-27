import torch
import time

def cuda_benchmark(device_id, N=1000000):
    # 指定要使用的显卡设备
    torch.cuda.set_device(device_id)

    # 创建输入数据
    data = torch.ones(N).cuda()

    # 启动CUDA操作，并记录执行时间
    start_time = time.time()
    for i in range(10000):
        data += 1
    torch.cuda.synchronize()  # 等待CUDA操作执行完成
    end_time = time.time()

    # 将结果从GPU内存下载到主机内存
    result = data.cpu().numpy()

    # 打印Benchmark结果和执行时间
    print(f"Benchmark结果：{result[:10]}")
    print(f"执行时间：{end_time - start_time} 秒")


if __name__ == '__main__':
    # 测试第一块显卡
    device_id = 0
    cuda_benchmark(device_id,10000000)
