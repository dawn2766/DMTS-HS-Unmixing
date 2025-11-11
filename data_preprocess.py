"""
Jasper Ridge数据集预处理模块
论文章节：Section 4.1 - Dataset Description
"""

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import os
import urllib.request

class JasperRidgePreprocessor:
    """Jasper Ridge数据集预处理器"""
    
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
    def download_data(self):
        """下载Jasper Ridge数据集"""
        url = "http://www.ehu.eus/ccwintco/uploads/2/22/Jasper.mat"
        save_path = os.path.join(self.data_path, 'JasperRidge.mat')
        
        if not os.path.exists(save_path):
            print(f"Downloading Jasper Ridge dataset from {url}...")
            urllib.request.urlretrieve(url, save_path)
            print("Download completed!")
        else:
            print("Dataset already exists.")
        
        return save_path
    
    def load_and_preprocess(self, mat_file=None):
        """
        加载并预处理Jasper Ridge数据
        返回：处理后的2D数据(N×L)和真实端元/丰度(如果有)
        """
        if mat_file is None:
            mat_file = self.download_data()
        
        print("Loading Jasper Ridge dataset...")
        data = sio.loadmat(mat_file)
        
        # 读取数据（假设键名为'Y'或'jasperRidge2_R198'）
        if 'Y' in data:
            hsi_data = data['Y']
        elif 'jasperRidge2_R198' in data:
            hsi_data = data['jasperRidge2_R198']
        else:
            # 尝试读取第一个非元数据键
            keys = [k for k in data.keys() if not k.startswith('__')]
            hsi_data = data[keys[0]]
        
        print(f"Original data shape: {hsi_data.shape}")
        
        # 数据已经是2D格式 (L, N) - 波段×像素
        if len(hsi_data.shape) == 2:
            L, N = hsi_data.shape
            print(f"Data is already in 2D format: L={L} bands, N={N} pixels")
            
            # 转置为 (N×L) 格式
            hsi_2d = hsi_data.T  # (N×L)
            
            # 归一化到[0,1]
            hsi_2d = hsi_2d.astype(np.float32)
            min_val = np.min(hsi_2d)
            max_val = np.max(hsi_2d)
            hsi_2d = (hsi_2d - min_val) / (max_val - min_val + 1e-8)
            print(f"Normalized to [0,1]: min={np.min(hsi_2d):.4f}, max={np.max(hsi_2d):.4f}")
            
            print(f"Final shape: {hsi_2d.shape} (N={N}, L={L})")
            
            # 读取真实端元和丰度（如果存在）
            endmembers_gt = None
            abundances_gt = None
            
            if 'M' in data:
                endmembers_gt = data['M'].astype(np.float32)
                print(f"Ground truth endmembers shape: {endmembers_gt.shape}")
            
            if 'A' in data:
                abundances_gt = data['A'].astype(np.float32)
                # 确保丰度是 (N×P) 格式
                if abundances_gt.shape[0] != N:
                    abundances_gt = abundances_gt.T
                print(f"Ground truth abundances shape: {abundances_gt.shape}")
            
            # 保存预处理报告
            self.save_preprocess_report(hsi_2d.shape, L)
            
            return hsi_2d, endmembers_gt, abundances_gt
        
        else:
            raise ValueError(f"Unexpected data shape: {hsi_data.shape}. Expected 2D array (L, N).")
    
    def save_preprocess_report(self, final_shape, num_bands):
        """保存预处理报告"""
        report_path = os.path.join(self.data_path, 'preprocess_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Jasper Ridge数据预处理报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"最终数据维度: {final_shape}\n")
            f.write(f"有效波段数: {num_bands}\n")
            f.write(f"像素总数: {final_shape[0]}\n")
            f.write(f"归一化范围: [0, 1]\n")
            f.write(f"数据格式: (N×L) - 像素×波段\n")
        print(f"Preprocessing report saved to {report_path}")


class JasperRidgeDataset(Dataset):
    """Jasper Ridge PyTorch Dataset"""
    
    def __init__(self, data_2d):
        """
        Args:
            data_2d: (N×L) numpy array
        """
        self.data = torch.from_numpy(data_2d).float()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # 测试预处理
    preprocessor = JasperRidgePreprocessor()
    hsi_2d, endmembers_gt, abundances_gt = preprocessor.load_and_preprocess()
    
    # 创建Dataset
    dataset = JasperRidgeDataset(hsi_2d)
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Sample shape: {dataset[0].shape}")
