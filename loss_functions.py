"""
损失函数实现
论文章节：Section 3.3 - Loss Functions
包含：log(SAD)损失（EE网络）、SAD损失（丰度网络）、RMSE评估
"""

import torch
import torch.nn as nn
import numpy as np

def spectral_angle_distance(y_pred, y_true, eps=1e-8):
    """
    计算光谱角距离（SAD）
    论文公式：SAD = arccos(<y_pred, y_true> / (||y_pred|| * ||y_true||))
    
    Args:
        y_pred: (B×L) 预测光谱
        y_true: (B×L) 真实光谱
        eps: 数值稳定项
    Returns:
        sad: (B,) 每个样本的SAD值（弧度）
    """
    # 归一化
    y_pred_norm = torch.nn.functional.normalize(y_pred, p=2, dim=1, eps=eps)
    y_true_norm = torch.nn.functional.normalize(y_true, p=2, dim=1, eps=eps)
    
    # 计算余弦相似度
    cos_sim = torch.sum(y_pred_norm * y_true_norm, dim=1)
    cos_sim = torch.clamp(cos_sim, -1 + eps, 1 - eps)  # 防止数值误差
    
    # 计算SAD（弧度）
    sad = torch.acos(cos_sim)
    
    return sad


class LogSADLoss(nn.Module):
    """
    对数SAD损失（用于EE网络）
    修正：使用 log(1 + SAD) 避免负值
    """
    
    def __init__(self, eps=1e-8):
        super(LogSADLoss, self).__init__()
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B×L) 预测光谱
            y_true: (B×L) 真实光谱
        Returns:
            loss: 标量损失值
        """
        sad = spectral_angle_distance(y_pred, y_true, self.eps)
        
        # 使用 log(1 + SAD) 避免负值，同时保持单调性
        log_sad = torch.log(1.0 + sad)
        
        # 平均损失
        loss = torch.mean(log_sad)
        
        return loss


class SADLoss(nn.Module):
    """
    SAD损失（用于丰度网络）
    论文公式 (1-97)：L2 = (1/n) * Σ SAD(yi, ŷi)
    """
    
    def __init__(self, eps=1e-8):
        super(SADLoss, self).__init__()
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B×L) 预测光谱
            y_true: (B×L) 真实光谱
        Returns:
            loss: 标量损失值
        """
        sad = spectral_angle_distance(y_pred, y_true, self.eps)
        
        # 平均损失
        loss = torch.mean(sad)
        
        return loss


def compute_endmember_sad(M_pred, M_true, eps=1e-8):
    """
    计算端元SAD（用于评估）
    
    Args:
        M_pred: (L×P) 预测端元矩阵
        M_true: (L×P) 真实端元矩阵
        eps: 数值稳定项
    Returns:
        sad_per_endmember: (P,) 每个端元的SAD值
        mean_sad: 平均SAD（mSAD）
    """
    P = M_pred.shape[1]
    sad_values = []
    
    for p in range(P):
        m_pred = M_pred[:, p].unsqueeze(0)  # (1×L)
        m_true = M_true[:, p].unsqueeze(0)  # (1×L)
        
        sad = spectral_angle_distance(m_pred, m_true, eps)
        sad_values.append(sad.item())
    
    sad_values = np.array(sad_values)
    mean_sad = np.mean(sad_values)
    
    return sad_values, mean_sad


def compute_abundance_rmse(A_pred, A_true):
    """
    计算丰度RMSE（用于评估）
    
    Args:
        A_pred: (N×P) 预测丰度矩阵
        A_true: (N×P) 真实丰度矩阵
    Returns:
        rmse_per_endmember: (P,) 每个端元的RMSE值
        mean_rmse: 平均RMSE（mRMSE）
    """
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    
    P = A_pred.shape[1]
    rmse_values = []
    
    for p in range(P):
        mse = np.mean((A_pred[:, p] - A_true[:, p])**2)
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)
    
    rmse_values = np.array(rmse_values)
    mean_rmse = np.mean(rmse_values)
    
    return rmse_values, mean_rmse


if __name__ == "__main__":
    # 测试损失函数
    torch.manual_seed(42)
    
    # 模拟数据
    B, L, P = 32, 198, 4
    y_true = torch.rand(B, L)
    y_pred = y_true + torch.randn(B, L) * 0.1
    
    # 测试SAD损失
    sad_loss_fn = SADLoss()
    loss_sad = sad_loss_fn(y_pred, y_true)
    print(f"SAD Loss: {loss_sad.item():.4f}")
    
    # 测试log(SAD)损失
    log_sad_loss_fn = LogSADLoss()
    loss_log_sad = log_sad_loss_fn(y_pred, y_true)
    print(f"Log(SAD) Loss: {loss_log_sad.item():.4f}")
    
    # 测试端元SAD
    M_true = torch.rand(L, P)
    M_pred = M_true + torch.randn(L, P) * 0.05
    sad_values, mean_sad = compute_endmember_sad(M_pred, M_true)
    print(f"\nEndmember SAD: {sad_values}")
    print(f"Mean SAD (mSAD): {mean_sad:.4f}")
    
    # 测试丰度RMSE
    A_true = torch.rand(100, P)
    A_true = A_true / A_true.sum(dim=1, keepdim=True)
    A_pred = A_true + torch.randn(100, P) * 0.05
    A_pred = torch.clamp(A_pred, 0, 1)
    A_pred = A_pred / A_pred.sum(dim=1, keepdim=True)
    
    rmse_values, mean_rmse = compute_abundance_rmse(A_pred, A_true)
    print(f"\nAbundance RMSE: {rmse_values}")
    print(f"Mean RMSE (mRMSE): {mean_rmse:.4f}")
