import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from tcnn import layers as tlayers
import torchvision.models as models
from tcnn.applications.vision import models as t_models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizers_dict_medmnist(model_dict):
    """
    Get a dictionary of optimizers for the models.
    """
    return {
        model_name: torch.optim.Adam(model.parameters(),lr=0.001)
        for model_name, model in model_dict.items()
    }


def get_optimizers_medmnist(model):
    """
    Get an optimizer for the model.
    """
    return torch.optim.Adam(model.parameters(), lr=0.001)

def get_schedulers_dict_medmnist(optimizers_dict):
    """
    Get a dictionary of schedulers for the optimizers.
    """
    return {
        model_name: MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        for model_name, optimizer in optimizers_dict.items()
    }

def get_schedulers_medmnist(optimizer):
    """
    Get a scheduler for the optimizer.
    """
    return MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

class UnsupportedLayerError(Exception):
    """自定义异常，用于处理不支持的层类型"""
    pass

def count_operations_in_model(model, inputs):
    # 初始化计数器
    mul_count = 0
    add_count = 0
    compare_count = 0  # 用于统计最大值和最小值操作的次数
    linear_layers_mul_count = 0
    linear_layers_add_count = 0

    # 自定义钩子函数
    def count_operations(module, input, output):
        nonlocal mul_count, add_count, compare_count, linear_layers_mul_count, linear_layers_add_count

        if isinstance(module, nn.Linear):
            # 线性层中的乘法和加法
            input_features = module.in_features
            output_features = module.out_features
            if isinstance(input, tuple):
                input = input[0] 
            batch_size = input.size()[0]
            
            # 计算乘法操作次数
            mul_count += batch_size * input_features * output_features
            
            # 根据是否有 bias 来计算加法操作次数
            if module.bias is not None:
                add_count += batch_size * input_features * output_features
            else:
                add_count += batch_size * input_features * (output_features - 1)
            
            linear_layers_mul_count += batch_size * input_features * output_features
            linear_layers_add_count += add_count

        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # 卷积层中的乘法和加法
            input_channels = module.in_channels
            output_channels = module.out_channels
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
            num_elements_per_kernel = torch.prod(torch.tensor(kernel_size)).item()
            output_elements = output.numel()

            mul_count += num_elements_per_kernel * input_channels * output_channels * output_elements
            add_count += (input_channels - 1) * output_channels * output_elements * (num_elements_per_kernel - 1)
            if module.bias is not None:
                add_count += output_channels * output_elements

        elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            # 平均池化层中的加法和除法
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
            num_elements_per_kernel = torch.prod(torch.tensor(kernel_size)).item()
            output_elements = output.numel()
            add_count += output_elements * (num_elements_per_kernel - 1)
            mul_count += output_elements  # 对应平均池化的除法操作
        
        elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            # 最大池化层中的比较操作
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
            num_elements_per_kernel = torch.prod(torch.tensor(kernel_size)).item()
            output_elements = output.numel()
            compare_count += (num_elements_per_kernel - 1) * output_elements

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 批归一化层中的加法和乘法
            output_elements = output.numel()
            mul_count += output_elements * 2  # 一个乘法用于归一化，一个乘法用于缩放
            add_count += output_elements * 2  # 一个加法用于归一化，一个加法用于平移
        elif isinstance(module, (nn.ReLU, nn.ReLU6)):
            # ReLU激活函数中的比较操作
            output_elements = output.numel()
            compare_count += output_elements
        
        elif isinstance(module, (nn.Sigmoid, nn.Tanh)):
            # Sigmoid和Tanh激活函数中的乘法和加法
            output_elements = output.numel()
            mul_count += output_elements
            add_count += output_elements

        elif isinstance(module,(tlayers.Conv1d, tlayers.Conv2d, tlayers.Conv3d)):
            # 卷积层中的乘法和加法
            input_channels = module.in_channels
            output_channels = module.out_channels
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
            num_elements_per_kernel = torch.prod(torch.tensor(kernel_size)).item()

            output_elements = output.numel()
            mul_count += input_channels * output_channels * num_elements_per_kernel * output_elements
            add_count += (input_channels * output_channels - 1) * output_elements

            if module.bias is not None:
                add_count += output_channels * output_elements
            # Tropical Convolutional Layers
        elif isinstance(module, (tlayers.MaxPlusMaxConv1d, tlayers.MaxPlusMaxConv2d, tlayers.MaxPlusMaxConv3d,
                                 tlayers.MaxPlusMinConv1d, tlayers.MaxPlusMinConv2d, tlayers.MaxPlusMinConv3d,
                                 tlayers.MinPlusMaxConv1d, tlayers.MinPlusMaxConv2d, tlayers.MinPlusMaxConv3d,
                                 tlayers.MinPlusMinConv1d, tlayers.MinPlusMinConv2d, tlayers.MinPlusMinConv3d)):
            # Max-Plus卷积层中的最大值和加法
            input_channels = module.in_channels
            output_channels = module.out_channels
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
            num_elements_per_kernel = torch.prod(torch.tensor(kernel_size)).item()

            output_elements = output.numel()
            add_count += num_elements_per_kernel * input_channels * output_channels * output_elements
            compare_count += (num_elements_per_kernel * input_channels - 1) * output_channels * output_elements

            if module.bias is not None:
                add_count += output_channels * output_elements

        
        elif isinstance(module, (tlayers.MaxPlusSumConv1d, tlayers.MaxPlusSumConv2d, tlayers.MaxPlusSumConv3d,
                                 tlayers.MinPlusSumConv1d, tlayers.MinPlusSumConv2d, tlayers.MinPlusSumConv3d)):
            # Max-Plus卷积层中的最大值和加法
            input_channels = module.in_channels
            output_channels = module.out_channels
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
            num_elements_per_kernel = torch.prod(torch.tensor(kernel_size)).item()

            output_elements = output.numel()
            add_count += (input_channels -1 + num_elements_per_kernel * input_channels) * output_channels * output_elements
            compare_count += (num_elements_per_kernel - 1) * input_channels * output_channels * output_elements

            if module.bias is not None:
                add_count += output_channels * output_elements

        elif isinstance(module, (tlayers.CompoundMinMaxPlusSumConv1d1p, tlayers.CompoundMinMaxPlusSumConv1d2p, tlayers.CompoundMinMaxPlusSumConv2d1p, tlayers.CompoundMinMaxPlusSumConv2d2p, tlayers.CompoundMinMaxPlusSumConv3d1p, tlayers.CompoundMinMaxPlusSumConv3d2p)):
            # 复合卷积层中的最大值、最小值和加法
            input_channels = module.in_channels
            output_channels = module.out_channels
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
            num_elements_per_kernel = torch.prod(torch.tensor(kernel_size)).item()

            output_elements = output.numel()
            mul_count += 2 * input_channels * output_channels * output_channels
            add_count += (num_elements_per_kernel * input_channels + 2 * input_channels -1 ) * output_channels * output_elements
            compare_count += 2 * (num_elements_per_kernel - 1) * input_channels * output_channels * output_elements

            if module.bias is not None:
                add_count += output_channels * output_elements

        elif isinstance(module, (tlayers.ParallelMinMaxPlusSumConv1d1p, tlayers.ParallelMinMaxPlusSumConv1d2p, tlayers.ParallelMinMaxPlusSumConv2d1p, tlayers.ParallelMinMaxPlusSumConv2d2p, tlayers.ParallelMinMaxPlusSumConv3d1p, tlayers.ParallelMinMaxPlusSumConv3d2p)):
            # 并行卷积层中的最大值、最小值和加法
            input_channels = module.in_channels
            output_channels = module.out_channels
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
            num_elements_per_kernel = torch.prod(torch.tensor(kernel_size)).item()

            output_elements = output.numel()
            mul_count += 2 * input_channels * output_channels * output_channels
            add_count += (2*num_elements_per_kernel * input_channels + 2 * input_channels -1 ) * output_channels * output_elements
            compare_count += 2 * (num_elements_per_kernel - 1) * input_channels * output_channels * output_elements

            if module.bias is not None:
                add_count += output_channels * output_elements

        elif isinstance(module, nn.Flatten):
            pass  # Flatten层不包含任何操作

        elif isinstance(module, nn.Module):
            # 递归处理模块中的子模块
            children = list(module.children())  # 将生成器转换为列表
            for child in children:
                # 子模块输入与当前模块的输入或输出一致
                child_input = input if child is children[-1] else output
                count_operations(child, child_input, output)

        else:
            # 遇到不支持的层类型，抛出异常
            raise UnsupportedLayerError(f"Unsupported layer type: {type(module).__name__}")
        

    # 注册钩子
    def register_hooks(layer):
            hooks = []
            for child in layer.children():
                if isinstance(child, nn.Sequential):
                    hooks.extend(register_hooks(child))
                else:
                    hook = child.register_forward_hook(count_operations)
                    hooks.append(hook)
            return hooks

    # 注册钩子
    hooks = register_hooks(model)

    # 执行前向传播
    try:
        output = model(inputs)
    except UnsupportedLayerError as e:
        print(e)
        # 移除钩子并终止程序
        for hook in hooks:
            hook.remove()
        return None

    # 移除钩子
    for hook in hooks:
        hook.remove()

    return mul_count, add_count, compare_count, linear_layers_mul_count, linear_layers_add_count


def count_residual_additions(model, input_tensor):
    residual_add_count = 0

    def count_residual_addition(module, input, output, name):
        nonlocal residual_add_count
        if hasattr(module, 'downsample') or isinstance(module, (models.resnet.BasicBlock, models.resnet.Bottleneck, 
                                                                t_models.Compound2pBasicBlock1, t_models.Compound2pBasicBlock2,
                                                                t_models.Parallel2pBasicBlock1,t_models.Parallel2pBasicBlock2)):
            # add_ops = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3] 
            residual_add_count += output.numel()  # 计算残差块中的加法操作次数
            # assert add_ops == output.numel()
            # print(f"Module: {name}, Residual Addition Count: {output.numel()} Output Shape: {output.shape}")

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (models.resnet.BasicBlock, models.resnet.Bottleneck, 
                            t_models.Compound2pBasicBlock1, t_models.Compound2pBasicBlock2,
                            t_models.Parallel2pBasicBlock1,t_models.Parallel2pBasicBlock2)):
            hooks.append(layer.register_forward_hook(
                lambda module, input, output, name=name: count_residual_addition(module, input, output, name)
            ))

    with torch.no_grad():
        _ = model(input_tensor)  # 执行前向传播

    for hook in hooks:
        hook.remove()

    return residual_add_count

if __name__ == '__main__':
    # 使用示例
    class TropicalLeNet2(nn.Module):
        def __init__(
            self,
            input_channels,
            num_classes,
            linear_size=880,
            first_layer_kernel_size=80,
            second_layer_kernel_size=3,
        ):
            super().__init__()

            self.conv1 = nn.Conv1d(
                input_channels, 6, kernel_size=first_layer_kernel_size, padding=2
            )
            self.conv2 = tlayers.MaxPlusMinConv1d(
                6, 16, kernel_size=second_layer_kernel_size
            )

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(linear_size, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

            self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, input):
            
            x = self.avg_pool(self.conv1(input))
            x = self.avg_pool(self.conv2(x))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = TropicalLeNet2(num_classes=10, input_channels=1)
    inputs = torch.randn(4, 1, 300)
    print(model)

    result = count_operations_in_model(model, inputs)

    if result:
        mul_count, add_count, compare_count, linear_layers_mul_count, linear_layers_add_count = result
        print(f"乘法次数: {mul_count}")
        print(f"加法次数: {add_count}")
        print(f"比较操作次数（最大值/最小值）: {compare_count}")

