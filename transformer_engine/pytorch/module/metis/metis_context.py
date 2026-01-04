from contextlib import contextmanager
import copy
from functools import cache

class LinearLowbitContext:
    
    # === 静态配置/成员变量 ===
    q_forward_input = "Cast2Fp4e2m1"
    q_forward_weight = "Cast2Fp4e2m1"
    q_backward_input = "Cast2Fp4e2m1"
    q_backward_weight = "Cast2Fp4e2m1"
    q_backward_outputgrad = "Cast2Fp4e2m1"

    # SVD & low-rank 配置
    activation_lowrank_niter = 2
    backward_lowrank_niter = 2
    q_scalar = 1.0
    enable_activation_svd = False
    activation_lowrank_svd = -1
    enable_backward_svd = False
    backward_lowrank_svd = -1
    activation_broadcast_dim = -1
    backward_broadcast_dim = -1
    activation_longtail_schedule = "none"
    backward_longtail_schedule = "none"
    enable_lowbit = True
    forward_svd_rank = -1
    enable_weight_svd = False
    gradacc_broadcast = False
    separate_residual_quantization  = True

    # 动态改变的参数
    load_history = False
    use_metis = True

    @classmethod
    def get_params(cls):
        """
        核心方法：动态获取当前实例所有的成员变量名称和值。
        过滤掉以 '_' 开头的私有属性和可调用的方法。
        """
        params = {}
        # dir(self) 获取包括类属性和实例属性在内的所有属性名称
        for name in dir(cls):
            # 1. 过滤掉魔术方法 (__init__等) 和 私有变量 (_variable)
            if name.startswith("_"):
                continue
            
            value = getattr(cls, name)
            
            # 2. 过滤掉方法 (def function)，只保留数据
            if callable(value):
                continue
                
            params[name] = value
        return params

    @classmethod
    @cache
    def get_params_names(cls):
        """
        核心方法：动态获取当前实例所有的成员变量名称。
        过滤掉以 '_' 开头的私有属性和可调用的方法。
        """
        params = []
        # dir(self) 获取包括类属性和实例属性在内的所有属性名称
        for name in dir(cls):
            # 1. 过滤掉魔术方法 (__init__等) 和 私有变量 (_variable)
            if name.startswith("_"):
                continue
            
            value = getattr(cls, name)
            
            # 2. 过滤掉方法 (def function)，只保留数据
            if callable(value):
                continue
                
            params.append(name)
        return params

    def __repr__(self) -> str:
        """
        完全自动化的 repr，不需要手动添加新变量。
        """
        def fmt_val(v):
            # 如果值是函数类对象（不太可能，但在你的原代码中有处理），返回名字
            if callable(v):
                return v.__name__
            # 字符串需要显式带引号，repr() 会自动处理
            return repr(v)

        # 获取所有参数
        params = self.get_params()
        
        # 生成格式化字符串列表
        param_strs = [f"  {k}={fmt_val(v)}" for k, v in params.items()]
        
        return f"{self.__class__.__name__}(\n" + ",\n".join(param_strs) + "\n)"

    def clone(self):
        """
        使用 get_params 实现的通用 clone，或者直接使用 deepcopy
        """
        new_obj = self.__class__()
        # 获取当前所有有效参数并赋值给新对象
        for k, v in self.get_params().items():
            setattr(new_obj, k, copy.deepcopy(v))
        return new_obj

# 测试函数：获取参数
def get_metis_context_param_names():
    """
    外部调用此函数获取字典格式的参数
    """
    return LinearLowbitContext.get_params_names()

@contextmanager
def get_metis_context(**kwargs):
    """
    用于临时修改 LinearLowbitContext 全局配置的上下文管理器。
    进入时按 kwargs 修改，退出时自动恢复。

    示例：
        with get_metis_context(q_scalar=0.5, enable_lowbit=False):
            # 临时使用低比特关闭配置
            ...
    """
    old_state = {}
    # print("entering metis context with ", kwargs)
    try:
        # 保存旧值并设置新值
        for key, value in kwargs.items():
            if hasattr(LinearLowbitContext, key):
                old_state[key] = getattr(LinearLowbitContext, key)
                setattr(LinearLowbitContext, key, value)
            else:
                raise AttributeError(f"LinearLowbitContext has no attribute '{key}'")
        # debugpy.breakpoint()
        yield
    finally:
        # 恢复原始值
        # print("exiting metis context with ", old_state)
        for key, value in old_state.items():
            setattr(LinearLowbitContext, key, value)

@contextmanager
def update_context(key,value):
    old_value = None
    try:
        if hasattr(LinearLowbitContext, key):
            old_value = getattr(LinearLowbitContext, key)
            setattr(LinearLowbitContext, key, value)
        else:
            raise AttributeError(f"LinearLowbitContext has no attribute '{key}'") 
        yield
    finally:
        assert old_value is not None,f"LinearLowbitContext key:{key}, value should not be {old_value}"
        setattr(LinearLowbitContext,key,old_value)

def load_svd_history():
    return update_context("load_history", True)

def no_use_metis():
    return update_context("use_metis", False)
    
