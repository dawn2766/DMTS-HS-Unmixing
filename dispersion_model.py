"""
散射模型（Dispersion Model）实现
论文章节：Section 2 - Dispersion Model for Spectral Variability
生成变异端元 M_var
"""

import torch
import torch.nn as nn
import numpy as np

class DispersionModel(nn.Module):
    """
    散射模型，基于质量弹簧方程和物理参数生成光谱变异
    论文公式 (1)-(5)
    """
    
    def __init__(self, n_bands=198, n_endmembers=4, K=3):
        """
        Args:
            n_bands: 波段数 L=198
            n_endmembers: 端元数 P=4
            K: 质量弹簧方程数（论文1-184: K=3）
        """
        super(DispersionModel, self).__init__()
        
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        self.K = K
        
        # 波长范围 380-2500 nm (AVIRIS)
        # 注册为buffer，这样会自动跟随模型移动到正确的设备
        self.register_buffer('wavelengths', torch.linspace(380, 2500, n_bands))
        
        # 可学习的散射参数 Λ=[ρ, ω0, γ, εr]
        # ρ: 粒子密度 [0.001, 3]
        self.rho = nn.Parameter(torch.rand(n_endmembers, K) * 2.999 + 0.001)
        
        # ω0: 共振频率（对应波长范围）
        self.omega0 = nn.Parameter(torch.rand(n_endmembers, K) * (2500-380) + 380)
        
        # γ: 阻尼系数 [0.001, 3]
        self.gamma = nn.Parameter(torch.rand(n_endmembers, K) * 2.999 + 0.001)
        
        # εr: 相对介电常数 [1, 8]
        self.epsilon_r = nn.Parameter(torch.rand(n_endmembers) * 7 + 1)
        
    def clamp_parameters(self):
        """约束参数在物理有效范围内"""
        with torch.no_grad():
            self.rho.clamp_(0.001, 3)
            self.omega0.clamp_(380, 2500)
            self.gamma.clamp_(0.001, 3)
            self.epsilon_r.clamp_(1, 8)
    
    def compute_susceptibility(self):
        """
        计算电磁感应率 χ(ω)
        论文公式 (1)-(2)
        """
        # wavelengths现在是buffer，会自动在正确的设备上
        omega = self.wavelengths.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        omega0 = self.omega0.unsqueeze(-1)  # (P, K, 1)
        
        # 计算每个质量弹簧的贡献
        # χ_k = ρ_k / (ω0_k^2 - ω^2 + i*γ_k*ω)
        rho = self.rho.unsqueeze(-1)  # (P, K, 1)
        gamma = self.gamma.unsqueeze(-1)  # (P, K, 1)
        
        # 实部和虚部
        denominator = (omega0**2 - omega**2)**2 + (gamma * omega)**2
        chi_real = rho * (omega0**2 - omega**2) / (denominator + 1e-8)
        chi_imag = rho * (gamma * omega) / (denominator + 1e-8)
        
        # 总感应率（K个弹簧求和）
        chi_real = chi_real.sum(dim=1)  # (P, L)
        chi_imag = chi_imag.sum(dim=1)  # (P, L)
        
        return chi_real, chi_imag
    
    def compute_refractive_index(self, chi_real):
        """
        计算折射率 n(ω)
        论文公式 (3): n = sqrt(εr + χ_real)
        """
        epsilon_r = self.epsilon_r.unsqueeze(-1)  # (P, 1)
        n = torch.sqrt(epsilon_r + chi_real + 1e-8)
        return n
    
    def compute_reflectance(self, n, chi_imag):
        """
        计算反射率 R(ω)
        论文公式 (4)-(5): Fresnel方程 + 吸收
        """
        # Fresnel反射系数（垂直入射）
        R_fresnel = ((n - 1)**2) / ((n + 1)**2 + 1e-8)
        
        # 吸收系数（基于虚部）
        absorption = torch.exp(-torch.abs(chi_imag))
        
        # 最终反射率
        R = R_fresnel * absorption
        
        return R
    
    def forward(self, M_vca):
        """
        生成变异端元
        Args:
            M_vca: (L×P) VCA提取的初始端元
        Returns:
            M_var: (L×P) 变异端元
        """
        self.clamp_parameters()
        
        # 计算散射特性
        chi_real, chi_imag = self.compute_susceptibility()
        n = self.compute_refractive_index(chi_real)
        R = self.compute_reflectance(n, chi_imag)
        
        # 限制反射率在合理范围内 [0, 1]
        R = torch.clamp(R, 0.0, 1.0)
        
        # 应用散射变化到初始端元
        # M_var = M_vca ⊙ R^T
        # 确保M_vca在正确的设备上
        if isinstance(M_vca, torch.Tensor):
            M_vca_torch = M_vca.to(self.wavelengths.device)
        else:
            M_vca_torch = torch.from_numpy(M_vca).float().to(self.wavelengths.device)
        
        # 使用加权方式而非直接相乘，避免产生过小的值
        # M_var = α * M_vca ⊙ R^T，其中α是可学习的缩放因子
        M_var = M_vca_torch * R.T  # (L×P)
        
        # 限制变异端元的范围，避免异常值
        M_var = torch.clamp(M_var, 0.0, 1.0)
        
        return M_var


if __name__ == "__main__":
    # 测试散射模型
    from data_preprocess import JasperRidgePreprocessor
    from vca import VCA
    
    # 加载数据
    preprocessor = JasperRidgePreprocessor()
    hsi_2d, _, _ = preprocessor.load_and_preprocess()
    
    # 提取初始端元
    vca = VCA(n_endmembers=4)
    M_vca = vca.fit(hsi_2d.T)
    
    # 应用散射模型
    dm = DispersionModel(n_bands=198, n_endmembers=4, K=3)
    M_var = dm(torch.from_numpy(M_vca).float())
    
    print(f"M_vca shape: {M_vca.shape}")
    print(f"M_var shape: {M_var.shape}")
    print(f"M_var range: [{M_var.min():.4f}, {M_var.max():.4f}]")
