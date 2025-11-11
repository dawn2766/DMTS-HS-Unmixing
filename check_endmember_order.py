"""
端元顺序检查脚本
用于分析VCA提取的端元与真实端元的对应关系
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from data_preprocess import JasperRidgePreprocessor
from vca import VCA
import os

def compute_correlation_matrix(M_vca, M_true):
    """
    计算VCA端元与真实端元之间的相关系数矩阵
    
    Args:
        M_vca: (L×P) VCA提取的端元
        M_true: (L×P) 真实端元
    
    Returns:
        corr_matrix: (P×P) 相关系数矩阵
    """
    P = M_vca.shape[1]
    corr_matrix = np.zeros((P, P))
    
    for i in range(P):
        for j in range(P):
            corr_matrix[i, j] = np.corrcoef(M_vca[:, i], M_true[:, j])[0, 1]
    
    return corr_matrix


def find_best_matching(corr_matrix):
    """
    根据相关系数矩阵找到最佳匹配
    
    Args:
        corr_matrix: (P×P) 相关系数矩阵
    
    Returns:
        matching: 字典，VCA端元索引 -> 真实端元索引
    """
    P = corr_matrix.shape[0]
    matching = {}
    used_true = set()
    
    # 贪心匹配：每次选择相关系数最大的配对
    for _ in range(P):
        max_corr = -1
        best_vca_idx = -1
        best_true_idx = -1
        
        for i in range(P):
            if i in matching:
                continue
            for j in range(P):
                if j in used_true:
                    continue
                if corr_matrix[i, j] > max_corr:
                    max_corr = corr_matrix[i, j]
                    best_vca_idx = i
                    best_true_idx = j
        
        matching[best_vca_idx] = best_true_idx
        used_true.add(best_true_idx)
    
    return matching


def main():
    print("="*60)
    print("Endmember Order Checker")
    print("="*60)
    
    # 加载数据
    preprocessor = JasperRidgePreprocessor()
    hsi_2d, M_true, A_true = preprocessor.load_and_preprocess()
    
    if M_true is None:
        print("Error: No ground truth endmembers found in dataset!")
        return
    
    # 提取VCA端元
    vca = VCA(n_endmembers=4)
    M_vca = vca.fit(hsi_2d.T)
    
    print(f"\nVCA Endmembers shape: {M_vca.shape}")
    print(f"True Endmembers shape: {M_true.shape}")
    
    # 计算相关系数矩阵
    corr_matrix = compute_correlation_matrix(M_vca, M_true)
    
    print("\nCorrelation Matrix (VCA vs True):")
    print("         True0   True1   True2   True3")
    for i in range(4):
        print(f"VCA{i}:  ", end="")
        for j in range(4):
            print(f"{corr_matrix[i, j]:6.3f}  ", end="")
        print()
    
    # 找到最佳匹配
    matching = find_best_matching(corr_matrix)
    
    print("\nBest Matching:")
    endmember_names = ['Tree', 'Water', 'Soil', 'Road']
    for vca_idx, true_idx in sorted(matching.items()):
        corr = corr_matrix[vca_idx, true_idx]
        print(f"  VCA Endmember {vca_idx} <-> True Endmember {true_idx} ({endmember_names[true_idx]}), "
              f"Correlation: {corr:.4f}")
    
    # 可视化端元光谱
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    wavelengths = np.linspace(380, 2500, M_vca.shape[0])
    
    for vca_idx in range(4):
        ax = axes[vca_idx]
        
        # 绘制VCA端元
        ax.plot(wavelengths, M_vca[:, vca_idx], 'b-', linewidth=2, label=f'VCA {vca_idx}')
        
        # 绘制对应的真实端元
        true_idx = matching[vca_idx]
        ax.plot(wavelengths, M_true[:, true_idx], 'r--', linewidth=2, 
               label=f'True {true_idx} ({endmember_names[true_idx]})')
        
        ax.set_xlabel('Wavelength (nm)', fontsize=10)
        ax.set_ylabel('Reflectance', fontsize=10)
        ax.set_title(f'VCA {vca_idx} vs True {true_idx} (Corr={corr_matrix[vca_idx, true_idx]:.3f})', 
                    fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = './results/endmember_matching.png'
    os.makedirs('./results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEndmember matching plot saved to {save_path}")
    
    # 保存匹配结果
    with open('./results/endmember_matching.txt', 'w', encoding='utf-8') as f:
        f.write("Endmember Matching Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("Correlation Matrix:\n")
        f.write("         True0   True1   True2   True3\n")
        for i in range(4):
            f.write(f"VCA{i}:  ")
            for j in range(4):
                f.write(f"{corr_matrix[i, j]:6.3f}  ")
            f.write("\n")
        
        f.write("\n\nBest Matching:\n")
        for vca_idx, true_idx in sorted(matching.items()):
            corr = corr_matrix[vca_idx, true_idx]
            f.write(f"VCA Endmember {vca_idx} <-> True Endmember {true_idx} ({endmember_names[true_idx]}), "
                   f"Correlation: {corr:.4f}\n")
    
    print("Matching results saved to ./results/endmember_matching.txt")
    
    # 返回匹配结果供其他脚本使用
    return matching


if __name__ == "__main__":
    main()
