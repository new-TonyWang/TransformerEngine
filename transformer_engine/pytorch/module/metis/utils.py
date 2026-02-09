import torch

class TensorOffloadManager:
    """
    Tensor CPU/GPU offload 管理器
    自动管理tensor在CPU和GPU之间的转移，用于节省GPU显存
    
    使用方法:
        manager = TensorOffloadManager()
        
        # 存储数据（自动offload到CPU）
        manager["svd_history"] = [tensor1, tensor2, tensor3]
        manager["token_drop_noise"] = noise_tensor
        
        # 读取数据（自动load到GPU）
        ker, ug_sg, v = manager["svd_history"]
        noise = manager["token_drop_noise"]
        
        # 检查是否存在
        if "svd_history" in manager:
            pass
            
        # 获取（带默认值）
        noise = manager.get("token_drop_noise", None)
        
        # 清空
        manager.clear()
    """
    
    def __init__(self, auto_offload: bool = True, pin_memory: bool = True):
        """
        Args:
            auto_offload: 是否自动将tensor offload到CPU（默认True）
            pin_memory: 是否使用pinned memory加速CPU-GPU传输（默认True）
        """
        self._storage = {}
        self._auto_offload = auto_offload
        self._pin_memory = pin_memory
        self._device_info = {}  # 存储原始设备信息
    
    def _offload_to_cpu(self, data):
        """将数据offload到CPU"""
        if data is None:
            return None
        
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                # 记录原始设备
                device = data.device
                # 转移到CPU
                if self._pin_memory:
                    cpu_data = data.to('cpu', non_blocking=True).pin_memory()
                else:
                    cpu_data = data.to('cpu', non_blocking=True)
                return cpu_data, device
            else:
                return data, None
        elif isinstance(data, (list, tuple)):
            result = []
            devices = []
            for item in data:
                offloaded_item, device = self._offload_to_cpu(item)
                result.append(offloaded_item)
                devices.append(device)
            return (result if isinstance(data, list) else tuple(result)), devices
        else:
            return data, None
    
    def _load_to_gpu(self, data, device_info):
        """将数据load到GPU"""
        if data is None:
            return None
        
        if isinstance(data, torch.Tensor):
            if device_info is not None and data.device.type == 'cpu':
                # 恢复到原始GPU设备
                return data.to(device_info, non_blocking=True)
            else:
                return data
        elif isinstance(data, (list, tuple)):
            if isinstance(device_info, (list, tuple)):
                result = []
                for item, dev in zip(data, device_info):
                    result.append(self._load_to_gpu(item, dev))
                return result if isinstance(data, list) else tuple(result)
            else:
                return data
        else:
            return data
    
    def __setitem__(self, key: str, value):
        """存储数据，自动offload到CPU"""
        if self._auto_offload:
            offloaded_data, device_info = self._offload_to_cpu(value)
            self._storage[key] = offloaded_data
            self._device_info[key] = device_info
        else:
            self._storage[key] = value
            self._device_info[key] = None
    
    def __getitem__(self, key: str):
        """读取数据，自动load到GPU"""
        if key not in self._storage:
            raise KeyError(f"Key '{key}' not found in TensorOffloadManager")
        
        data = self._storage[key]
        device_info = self._device_info.get(key, None)
        
        if self._auto_offload:
            return self._load_to_gpu(data, device_info)
        else:
            return data
    
    def get(self, key: str, default=None):
        """获取数据，如果不存在返回默认值"""
        if key not in self._storage:
            return default
        return self.__getitem__(key)
    
    def __contains__(self, key: str):
        """检查key是否存在"""
        return key in self._storage
    
    def clear(self):
        """清空所有存储的数据"""
        self._storage.clear()
        self._device_info.clear()
    
    def keys(self):
        """返回所有keys"""
        return self._storage.keys()
    
    def __len__(self):
        """返回存储的数据数量"""
        return len(self._storage)
    
    def __repr__(self):
        return f"TensorOffloadManager(keys={list(self._storage.keys())}, auto_offload={self._auto_offload})"


