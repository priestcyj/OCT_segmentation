import torch
import time


def occupy_gpu(gpu_index, percentage=0.9, check_interval=60):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        print(f"Using GPU {gpu_index}")

        # 获取可用显存信息
        total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
        reserved_memory = torch.cuda.memory_reserved(gpu_index)
        allocated_memory = torch.cuda.memory_allocated(gpu_index)
        free_memory = total_memory - reserved_memory - allocated_memory
        memory_to_allocate = free_memory * percentage
        tensor_size = int(memory_to_allocate // 4)  # 由于float32类型，每个元素占用4字节
        print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
        print(f"Free memory before allocation: {free_memory / (1024 ** 3):.2f} GB")
        print(f"Allocating {memory_to_allocate / (1024 ** 3):.2f} GB")

        # 分配张量占用显存
        tensor_list = []
        try:
            tensor = torch.empty(tensor_size, dtype=torch.float32, device=f'cuda:{gpu_index}')
            tensor_list.append(tensor)
            print(f"Successfully allocated {memory_to_allocate / (1024 ** 3):.2f} GB on GPU {gpu_index}")
        except RuntimeError as e:
            print(f"Failed to allocate memory: {e}")
            return

        # 保持占用显存
        try:
            while True:
                time.sleep(check_interval)
                allocated_memory = torch.cuda.memory_allocated(gpu_index)
                reserved_memory = torch.cuda.memory_reserved(gpu_index)
                free_memory = total_memory - reserved_memory - allocated_memory
                print(
                    f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB, Free memory: {free_memory / (1024 ** 3):.2f} GB")

                # 如果有显存释放，重新分配
                if free_memory > total_memory * (1 - percentage):
                    additional_tensor = torch.empty(tensor_size, dtype=torch.float32, device=f'cuda:{gpu_index}')
                    tensor_list.append(additional_tensor)
                    print(f"Reallocated additional memory, Free memory: {free_memory / (1024 ** 3):.2f} GB")
        except KeyboardInterrupt:
            print("Stopped by user.")
    else:
        print("No GPU available.")


# 占用第0张GPU的90%显存，并每60秒检查一次
occupy_gpu(gpu_index=2, percentage=0.8, check_interval=60)
