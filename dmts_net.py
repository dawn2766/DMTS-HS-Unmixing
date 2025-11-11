"""
DMTS-Net (Dual-stream Model with Twin Structure) 网络架构
论文章节：Section 3 - Proposed Method
包含端元提取网络（EE Network）和丰度估计自编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dispersion_model import DispersionModel

class EndmemberExtractionNetwork(nn.Module):
    """
    端元提取网络（EE Network）
    论文章节：Section 3.1
    架构：4层FC + LeakyReLU(α=0.4) + BN + 散射模型
    """
    
    def __init__(self, n_bands=198, n_endmembers=4, K=3, hidden_dims=[512, 256, 128, 64]):
        super(EndmemberExtractionNetwork, self).__init__()
        
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        
        # 4层全连接网络
        self.fc1 = nn.Linear(n_bands, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])
        
        # 参数提取层（输出散射模型参数）
        self.param_layer = nn.Linear(hidden_dims[3], n_endmembers * K * 4)  # 4参数
        
        # LeakyReLU激活函数（α=0.4）
        self.leaky_relu = nn.LeakyReLU(0.4)
        
        # 散射模型
        self.dispersion_model = DispersionModel(n_bands, n_endmembers, K)
        
    def forward(self, x, M_vca):
        """
        Args:
            x: (B×L) 输入光谱
            M_vca: (L×P) VCA初始端元
        Returns:
            M_hat: (L×P) 估计端元
        """
        # 4层全连接网络
        h = self.leaky_relu(self.bn1(self.fc1(x)))
        h = self.leaky_relu(self.bn2(self.fc2(h)))
        h = self.leaky_relu(self.bn3(self.fc3(h)))
        h = self.leaky_relu(self.bn4(self.fc4(h)))
        
        # 提取参数（不直接使用，散射模型参数自学习）
        # params = self.param_layer(h)
        
        # 使用散射模型生成变异端元
        M_var = self.dispersion_model(M_vca)
        
        # 合并端元：M_hat = M_var + M_vca
        M_hat = M_var + M_vca
        
        return M_hat


class AbundanceEstimationNetwork(nn.Module):
    """
    丰度估计自编码器网络
    论文章节：Section 3.2
    编码器：5层1D卷积 + 2层FC，激活函数Tanh
    解码器：1层FC（无激活/偏置）
    """
    
    def __init__(self, n_bands=198, n_endmembers=4):
        super(AbundanceEstimationNetwork, self).__init__()
        
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        
        # 编码器：5层1D卷积
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # 全局池化后的维度
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_endmembers)
        
        # 解码器：1层FC（无激活，无偏置）
        self.decoder = nn.Linear(n_endmembers, n_bands, bias=False)
        
        # Tanh激活
        self.tanh = nn.Tanh()
        
    def encode(self, x):
        """
        编码器
        Args:
            x: (B×L) 输入光谱
        Returns:
            A: (B×P) 丰度向量
        """
        # 添加通道维度 (B×1×L)
        x = x.unsqueeze(1)
        
        # 5层卷积
        h = self.tanh(self.conv1(x))
        h = self.tanh(self.conv2(h))
        h = self.tanh(self.conv3(h))
        h = self.tanh(self.conv4(h))
        h = self.tanh(self.conv5(h))
        
        # 全局平均池化
        h = torch.mean(h, dim=2)  # (B×256)
        
        # 2层全连接
        h = self.tanh(self.fc1(h))
        A = self.fc2(h)  # (B×P)
        
        # 丰度约束：非负 + 和为1
        A = torch.clamp(A, min=0)  # 非负约束
        A = F.softmax(A, dim=1)  # 和为1约束
        
        return A
    
    def decode(self, A, M_hat):
        """
        解码器：Y_hat = A @ M_hat^T
        Args:
            A: (B×P) 丰度矩阵
            M_hat: (L×P) 端元矩阵
        Returns:
            Y_hat: (B×L) 重构光谱
        """
        # Y_hat = A @ M_hat^T
        Y_hat = torch.mm(A, M_hat.T)  # (B×L)
        
        return Y_hat
    
    def forward(self, x, M_hat):
        """
        Args:
            x: (B×L) 输入光谱
            M_hat: (L×P) 估计端元
        Returns:
            A: (B×P) 丰度矩阵
            Y_hat: (B×L) 重构光谱
        """
        A = self.encode(x)
        Y_hat = self.decode(A, M_hat)
        
        return A, Y_hat


class DMTSNet(nn.Module):
    """
    完整的DMTS-Net模型
    双流架构：EE Network + Abundance Network
    """
    
    def __init__(self, n_bands=198, n_endmembers=4, K=3):
        super(DMTSNet, self).__init__()
        
        self.ee_network = EndmemberExtractionNetwork(n_bands, n_endmembers, K)
        self.abundance_network = AbundanceEstimationNetwork(n_bands, n_endmembers)
        
    def forward(self, x, M_vca, return_endmembers=False):
        """
        Args:
            x: (B×L) 输入光谱
            M_vca: (L×P) VCA初始端元
            return_endmembers: 是否返回端元
        Returns:
            A: (B×P) 丰度矩阵
            Y_hat: (B×L) 重构光谱
            M_hat: (L×P) 估计端元（可选）
        """
        # 端元提取
        M_hat = self.ee_network(x, M_vca)
        
        # 丰度估计
        A, Y_hat = self.abundance_network(x, M_hat)
        
        if return_endmembers:
            return A, Y_hat, M_hat
        else:
            return A, Y_hat


if __name__ == "__main__":
    # 测试DMTS-Net
    from data_preprocess import JasperRidgePreprocessor
    from vca import VCA
    
    # 加载数据
    preprocessor = JasperRidgePreprocessor()
    hsi_2d, _, _ = preprocessor.load_and_preprocess()
    
    # 提取VCA端元
    vca = VCA(n_endmembers=4)
    M_vca = torch.from_numpy(vca.fit(hsi_2d.T)).float()
    
    # 创建模型
    model = DMTSNet(n_bands=198, n_endmembers=4, K=3)
    
    # 前向传播测试
    x = torch.from_numpy(hsi_2d[:32]).float()  # Batch=32
    A, Y_hat, M_hat = model(x, M_vca, return_endmembers=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Abundances shape: {A.shape}")
    print(f"Reconstructed shape: {Y_hat.shape}")
    print(f"Endmembers shape: {M_hat.shape}")
    print(f"Abundance sum (should be 1): {A[0].sum():.4f}")
