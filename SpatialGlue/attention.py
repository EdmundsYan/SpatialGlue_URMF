import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, out_dim, reduction_ratio=16, pool_types='avg'):
        """
        参数：
          gate_channels: 期望的输入通道数（用于初始化 Conv1d，但在 forward 中会根据实际输入动态调整）
          out_dim: 希望的输出特征维度（即 gating 模块最后输出的维度，例如 128）
          reduction_ratio: 降维比例（目前未使用，可扩展为 MLP 版本）
          pool_types: 使用的池化方式，支持 'avg', 'max', 'lp', 'lse'（默认 'avg'）
        """
        super(ChannelGate, self).__init__()
        self.expected_channels = gate_channels  # 记录期望通道数
        self.out_dim = out_dim
        self.pool_types = pool_types

        # 初始化 Conv1d，输入和输出通道均设为 expected_channels
        self.con = nn.Conv1d(self.expected_channels, self.expected_channels,
                             kernel_size=3, stride=1, padding=1, bias=False)
        init.xavier_uniform_(self.con.weight)
        
        self.fc = None  # 后续用于将融合输出投影到 out_dim

    def forward(self, x):
        """
        输入：
            x: 期望形状为 [batch, C, L]，其中 C 为通道数，L 为特征长度；
               如果输入为 2D（形如 [batch, features]），会自动扩展为 [batch, 1, features]
        处理流程：
          1. 若输入为2D，则 unsqueeze 到 [batch, 1, L]
          2. 检查当前通道数，如果不匹配则重新初始化 Conv1d
          3. 根据 pool_types 对 x 在最后一维（L）进行池化，得到 shape [batch, C, 1]
          4. 经过 Conv1d 处理，再与池化结果相加并经过 Sigmoid 激活得到注意力权重
          5. 将注意力权重用于调整原输入，并采用残差连接
          6. 沿通道维度求和，得到形状 [batch, L] 的融合输出
          7. 如果 fc 未初始化，则根据融合输出的维度（L）初始化 fc，将输出投影到 out_dim
          8. 返回投影后的结果，其形状为 [batch, out_dim]
        """
        # 如果输入为2D，则扩展为3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 变为 [batch, 1, features]

        # 获取当前通道数
        current_channels = x.size(1)
        if current_channels != self.expected_channels:
            self.expected_channels = current_channels
            self.con = nn.Conv1d(current_channels, current_channels,
                                 kernel_size=3, stride=1, padding=1, bias=False).to(x.device)
            init.xavier_uniform_(self.con.weight)
        
        # 根据 pool_types 选择池化方式，在最后一维做池化
        if self.pool_types == 'avg':
            pool_out = x.mean(dim=2, keepdim=True)  # shape: [batch, C, 1]
        elif self.pool_types == 'max':
            pool_out = x.max(dim=2, keepdim=True)[0]  # shape: [batch, C, 1]
        elif self.pool_types == 'lp':
            pool_out = x.norm(2, dim=2, keepdim=True)  # L2范数池化
        elif self.pool_types == 'lse':
            pool_out = torch.logsumexp(x, dim=2, keepdim=True)  # LSE池化
        else:
            raise ValueError("Unsupported pool_types: {}".format(self.pool_types))
        
        # 将池化结果经过 Conv1d 得到中间特征
        channel_att_raw = self.con(pool_out)  # shape: [batch, C, 1]
        # 计算注意力权重：将池化结果与卷积输出相加后经过 Sigmoid
        score = torch.sigmoid(pool_out + channel_att_raw)  # shape: [batch, C, 1]
        
        # 将注意力权重作用于原输入，并加上残差连接
        out = x * score + x  # shape: [batch, C, L]
        # 融合：对通道维度求和，得到 shape [batch, L]
        fusion_output = out.sum(dim=1)
        
        # 如果 fc 未初始化，则根据融合输出的最后一维大小初始化 fc
        if self.fc is None:
            L = fusion_output.size(1)
            self.fc = nn.Linear(L, self.out_dim).to(x.device)
            init.xavier_uniform_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias, 0)
        
        fusion_output = self.fc(fusion_output)  # shape: [batch, out_dim]
        return fusion_output

if __name__ == "__main__":
    # 测试 3D 输入：假设输入形状为 [50, 3, 72]
    a = torch.randn(50, 3, 72).to('cuda')
    fusion_layer = ChannelGate(3, out_dim=128, reduction_ratio=3, pool_types='avg').to('cuda')
    b = fusion_layer(a)
    print("3D Input -> Output shape:", b.shape)  # 预期输出: [50, 128]
    
    # 测试 2D 输入：假设输入形状为 [50, 72]
    c = torch.randn(50, 72).to('cuda')
    d = fusion_layer(c)
    print("2D Input -> Output shape:", d.shape)  # 预期输出: [50, 128]