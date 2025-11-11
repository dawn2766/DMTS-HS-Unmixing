"""
DMTS-Net评估脚本
生成端元光谱图、丰度分布图、性能指标表
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader

from dmts_net import DMTSNet
from loss_functions import compute_endmember_sad, compute_abundance_rmse
from data_preprocess import JasperRidgeDataset

class DMTSEvaluator:
    """DMTS-Net评估器"""
    
    def __init__(self, model, M_vca, M_true=None, A_true=None, 
                 device='cuda', save_dir='./results'):
        self.model = model.to(device)
        self.M_vca = M_vca.to(device)
        self.M_true = M_true
        self.A_true = A_true
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 端元名称
        self.endmember_names = ['Tree', 'Water', 'Soil', 'Road']
        
        # 端元匹配索引（初始为None，在extract_results中计算）
        self.match_indices = None
    
    def match_endmembers(self, M_pred, M_true):
        """
        匹配预测端元和真实端元的顺序
        使用贪心算法：每次选择SAD最小的配对
        
        Args:
            M_pred: (L×P) 预测端元
            M_true: (L×P) 真实端元
        Returns:
            match_indices: (P,) 匹配索引，使得M_pred[:, match_indices[i]]对应M_true[:, i]
        """
        P = M_pred.shape[1]
        
        # 转换为torch张量
        if not isinstance(M_pred, torch.Tensor):
            M_pred = torch.from_numpy(M_pred).float()
        if not isinstance(M_true, torch.Tensor):
            M_true = torch.from_numpy(M_true).float()
        
        # 计算所有端元对之间的SAD
        sad_matrix = np.zeros((P, P))
        for i in range(P):
            for j in range(P):
                m_pred = M_pred[:, i].unsqueeze(0)
                m_true = M_true[:, j].unsqueeze(0)
                from loss_functions import spectral_angle_distance
                sad = spectral_angle_distance(m_pred, m_true)
                sad_matrix[i, j] = sad.item()
        
        # 贪心匹配：每次选择最小SAD且未被选择的配对
        match_indices = [-1] * P
        used_pred = [False] * P
        used_true = [False] * P
        
        # 按SAD从小到大排序所有配对
        pairs = []
        for i in range(P):
            for j in range(P):
                pairs.append((sad_matrix[i, j], i, j))
        pairs.sort()
        
        # 贪心选择
        for sad, i, j in pairs:
            if not used_pred[i] and not used_true[j]:
                match_indices[j] = i
                used_pred[i] = True
                used_true[j] = True
        
        print("\nEndmember Matching:")
        print("Predicted -> True (SAD)")
        for true_idx, pred_idx in enumerate(match_indices):
            sad_val = sad_matrix[pred_idx, true_idx]
            print(f"  Pred[{pred_idx}] -> True[{true_idx}] ({self.endmember_names[true_idx]}): SAD={sad_val:.4f}")
        
        return match_indices
        
    def extract_results(self, data_loader):
        """提取所有预测结果"""
        self.model.eval()
        
        all_A_pred = []
        M_hat = None
        
        with torch.no_grad():
            for x in data_loader:
                x = x.to(self.device)
                A, Y_hat, M = self.model(x, self.M_vca, return_endmembers=True)
                all_A_pred.append(A.cpu())
                if M_hat is None:
                    M_hat = M.cpu()
        
        A_pred = torch.cat(all_A_pred, dim=0).numpy()
        M_hat = M_hat.numpy()
        
        # 如果有真实端元，进行端元匹配
        if self.M_true is not None:
            M_true_np = self.M_true if isinstance(self.M_true, np.ndarray) else self.M_true.numpy()
            self.match_indices = self.match_endmembers(M_hat, M_true_np)
            
            # 重新排列预测端元和丰度，使其与真实端元对应
            M_hat = M_hat[:, self.match_indices]
            A_pred = A_pred[:, self.match_indices]
            
            print("\nEndmembers and abundances reordered to match ground truth.")
        
        return M_hat, A_pred
    
    def plot_endmembers(self, M_hat):
        """绘制端元光谱图（红色=真实，绿色=估计）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        wavelengths = np.linspace(380, 2500, M_hat.shape[0])
        
        for i in range(4):
            ax = axes[i]
            
            # 绘制估计端元（绿色）
            ax.plot(wavelengths, M_hat[:, i], 'g-', linewidth=2, label='Estimated')
            
            # 绘制真实端元（红色，如果有）
            if self.M_true is not None:
                M_true_np = self.M_true if isinstance(self.M_true, np.ndarray) else self.M_true.numpy()
                ax.plot(wavelengths, M_true_np[:, i], 'r--', linewidth=2, label='Ground Truth')
            
            ax.set_xlabel('Wavelength (nm)', fontsize=10)
            ax.set_ylabel('Reflectance', fontsize=10)
            ax.set_title(f'Endmember {i+1}: {self.endmember_names[i]}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'endmembers_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Endmember plots saved to {save_path}")
    
    def plot_abundances(self, A_pred):
        """绘制丰度分布图"""
        # 重塑为2D图像
        A_maps = A_pred.reshape(100, 100, 4)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i in range(4):
            im = axes[i].imshow(A_maps[:, :, i], cmap='jet', vmin=0, vmax=1)
            axes[i].set_title(f'Abundance Map: {self.endmember_names[i]}', 
                            fontsize=12, fontweight='bold')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'abundance_maps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Abundance maps saved to {save_path}")
        
        # 如果有真实丰度，绘制对比图
        if self.A_true is not None:
            A_true_np = self.A_true if isinstance(self.A_true, np.ndarray) else self.A_true.numpy()
            A_true_maps = A_true_np.reshape(100, 100, 4)
            
            fig, axes = plt.subplots(4, 2, figsize=(10, 16))
            
            for i in range(4):
                # 真实丰度
                im1 = axes[i, 0].imshow(A_true_maps[:, :, i], cmap='jet', vmin=0, vmax=1)
                axes[i, 0].set_title(f'{self.endmember_names[i]} - Ground Truth', fontsize=10)
                axes[i, 0].axis('off')
                plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
                
                # 估计丰度
                im2 = axes[i, 1].imshow(A_maps[:, :, i], cmap='jet', vmin=0, vmax=1)
                axes[i, 1].set_title(f'{self.endmember_names[i]} - Estimated', fontsize=10)
                axes[i, 1].axis('off')
                plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, 'abundance_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Abundance comparison saved to {save_path}")
    
    def compute_metrics(self, M_hat, A_pred):
        """计算并保存性能指标"""
        results = {}
        
        # 计算SAD（端元已经匹配过了）
        if self.M_true is not None:
            M_true_torch = torch.from_numpy(self.M_true if isinstance(self.M_true, np.ndarray) 
                                          else self.M_true.numpy()).float()
            M_hat_torch = torch.from_numpy(M_hat).float()
            
            sad_values, mean_sad = compute_endmember_sad(M_hat_torch, M_true_torch)
            
            results['sad_per_endmember'] = sad_values
            results['mean_sad'] = mean_sad
            
            print("\nEndmember SAD (after matching):")
            for i, name in enumerate(self.endmember_names):
                print(f"  {name}: {sad_values[i]:.4f}")
            print(f"  Mean SAD (mSAD): {mean_sad:.4f}")
            print(f"  Paper baseline: 0.0278")
        
        # 计算RMSE（丰度已经匹配过了）
        if self.A_true is not None:
            rmse_values, mean_rmse = compute_abundance_rmse(A_pred, self.A_true)
            
            results['rmse_per_endmember'] = rmse_values
            results['mean_rmse'] = mean_rmse
            
            print("\nAbundance RMSE (after matching):")
            for i, name in enumerate(self.endmember_names):
                print(f"  {name}: {rmse_values[i]:.4f}")
            print(f"  Mean RMSE (mRMSE): {mean_rmse:.4f}")
            print(f"  Paper baseline: 0.0568")
        
        # 保存结果到文件
        self.save_metrics_report(results)
        
        return results
    
    def save_metrics_report(self, results):
        """保存性能指标报告"""
        report_path = os.path.join(self.save_dir, 'performance_metrics.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DMTS-Net性能评估报告\n")
            f.write("="*60 + "\n\n")
            
            if 'sad_per_endmember' in results:
                f.write("端元提取性能 (SAD)\n")
                f.write("-"*60 + "\n")
                for i, name in enumerate(self.endmember_names):
                    f.write(f"{name:10s}: {results['sad_per_endmember'][i]:.6f}\n")
                f.write(f"{'Mean SAD':10s}: {results['mean_sad']:.6f}\n")
                f.write(f"{'Paper':10s}: 0.0278\n\n")
            
            if 'rmse_per_endmember' in results:
                f.write("丰度估计性能 (RMSE)\n")
                f.write("-"*60 + "\n")
                for i, name in enumerate(self.endmember_names):
                    f.write(f"{name:10s}: {results['rmse_per_endmember'][i]:.6f}\n")
                f.write(f"{'Mean RMSE':10s}: {results['mean_rmse']:.6f}\n")
                f.write(f"{'Paper':10s}: 0.0568\n")
        
        print(f"\nMetrics report saved to {report_path}")
    
    def run_full_evaluation(self, data_loader):
        """运行完整评估流程"""
        print("\n" + "="*60)
        print("Running Full Evaluation")
        print("="*60)
        
        # 提取结果
        M_hat, A_pred = self.extract_results(data_loader)
        
        # 绘制端元
        self.plot_endmembers(M_hat)
        
        # 绘制丰度
        self.plot_abundances(A_pred)
        
        # 计算指标
        results = self.compute_metrics(M_hat, A_pred)
        
        print("\nEvaluation completed!")
        
        return results


def main():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    from data_preprocess import JasperRidgePreprocessor
    from vca import VCA
    
    preprocessor = JasperRidgePreprocessor()
    hsi_2d, M_true, A_true = preprocessor.load_and_preprocess()
    
    # VCA提取初始端元
    vca = VCA(n_endmembers=4)
    M_vca = vca.fit(hsi_2d.T)
    M_vca_torch = torch.from_numpy(M_vca).float()
    
    # 创建数据集
    dataset = JasperRidgeDataset(hsi_2d)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 加载训练好的模型
    model = DMTSNet(n_bands=198, n_endmembers=4, K=3)
    model_path = './results/best_model.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # 创建评估器
    evaluator = DMTSEvaluator(
        model=model,
        M_vca=M_vca_torch,
        M_true=M_true,
        A_true=A_true,
        device=device,
        save_dir='./results'
    )
    
    # 运行评估
    results = evaluator.run_full_evaluation(data_loader)


if __name__ == "__main__":
    main()
