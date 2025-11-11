"""
VCA (Vertex Component Analysis) 算法实现
论文章节：Section 3.1 - Endmember Extraction
用于提取初始端元 M_vca
"""

import numpy as np
from sklearn.decomposition import PCA

class VCA:
    """VCA算法实现"""
    
    def __init__(self, n_endmembers):
        """
        Args:
            n_endmembers: 端元数量（Jasper Ridge P=4）
        """
        self.n_endmembers = n_endmembers
        
    def estimate_snr(self, Y):
        """估计信噪比"""
        h, w = Y.shape
        mean_Y = np.mean(Y, axis=1, keepdims=True)
        R_Y = Y - mean_Y
        
        # 信号功率
        P_signal = np.sum(Y**2) / w
        
        # 噪声功率（使用最小二乘估计）
        U, S, Vt = np.linalg.svd(R_Y, full_matrices=False)
        noise_subspace = U[:, self.n_endmembers:]
        P_noise = np.sum(S[self.n_endmembers:]**2) / w
        
        if P_noise > 0:
            snr = 10 * np.log10(P_signal / P_noise)
        else:
            snr = 0
        
        return max(snr, 0)
    
    def fit(self, Y):
        """
        执行VCA算法
        Args:
            Y: (L×N) 输入数据矩阵，L=波段数，N=像素数
        Returns:
            endmembers: (L×P) 端元矩阵
        """
        L, N = Y.shape
        
        print(f"Running VCA with {self.n_endmembers} endmembers...")
        
        # Step 1: 数据中心化
        mean_Y = np.mean(Y, axis=1, keepdims=True)
        Y_centered = Y - mean_Y
        
        # Step 2: 降维投影到 (P-1) 维子空间
        pca = PCA(n_components=self.n_endmembers - 1)
        Y_proj = pca.fit_transform(Y_centered.T).T  # (P-1×N)
        
        # Step 3: 投影到单纯形
        # 添加一维使数据位于单纯形上
        ones = np.ones((1, N))
        Y_simplex = np.vstack([Y_proj, ones])  # (P×N)
        
        # Step 4: 迭代查找端元（最大体积单纯形顶点）
        indices = []
        
        # 初始化：选择范数最大的像素
        norms = np.sum(Y_simplex**2, axis=0)
        max_idx = np.argmax(norms)
        indices.append(max_idx)
        
        # 迭代查找剩余端元
        for i in range(1, self.n_endmembers):
            # 计算当前单纯形的投影
            U = Y_simplex[:, indices]  # (P × len(indices))
            
            # 正交投影：proj_Y = Y - U * (U^T * U)^(-1) * U^T * Y
            # 修正：使用正确的投影公式
            if U.shape[1] == 1:
                # 只有一个端元时的特殊情况
                U_normalized = U / (np.linalg.norm(U) + 1e-8)
                proj_Y = Y_simplex - U_normalized @ (U_normalized.T @ Y_simplex)
            else:
                # 多个端元时使用QR分解求正交投影
                Q, R = np.linalg.qr(U)
                proj_Y = Y_simplex - Q @ (Q.T @ Y_simplex)
            
            # 选择投影后范数最大的像素
            norms = np.sum(proj_Y**2, axis=0)
            max_idx = np.argmax(norms)
            
            # 避免选择重复的端元
            if max_idx in indices:
                # 如果选到重复的，选择次大的
                sorted_indices = np.argsort(norms)[::-1]
                for idx in sorted_indices:
                    if idx not in indices:
                        max_idx = idx
                        break
            
            indices.append(max_idx)
        
        # Step 5: 从原始数据中提取端元
        endmembers = Y[:, indices]  # (L×P)
        
        print(f"VCA completed. Endmembers shape: {endmembers.shape}")
        print(f"Selected pixel indices: {indices}")
        
        return endmembers


if __name__ == "__main__":
    # 测试VCA
    from data_preprocess import JasperRidgePreprocessor
    
    preprocessor = JasperRidgePreprocessor()
    hsi_2d, _, _ = preprocessor.load_and_preprocess()
    
    # VCA需要(L×N)格式
    Y = hsi_2d.T  # (L×N)
    
    vca = VCA(n_endmembers=4)
    endmembers = vca.fit(Y)
    
    print(f"\nExtracted endmembers shape: {endmembers.shape}")
    print(f"Endmember values range: [{endmembers.min():.4f}, {endmembers.max():.4f}]")
