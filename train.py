"""
DMTS-Net训练脚本
论文章节：Section 4.2 - Training Strategy
分阶段训练：先训练EE网络 → 固定权重 → 训练丰度网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
from tqdm import tqdm

from dmts_net import DMTSNet
from loss_functions import LogSADLoss, SADLoss, compute_endmember_sad, compute_abundance_rmse
from data_preprocess import JasperRidgeDataset
from vca import VCA

class DMTSTrainer:
    """DMTS-Net训练器"""
    
    def __init__(self, model, M_vca, M_true=None, A_true=None, 
                 device='cuda', save_dir='./results'):
        """
        Args:
            model: DMTS-Net模型
            M_vca: (L×P) VCA初始端元
            M_true: (L×P) 真实端元（用于评估）
            A_true: (N×P) 真实丰度（用于评估）
            device: 训练设备
            save_dir: 结果保存目录
        """
        self.model = model.to(device)
        self.M_vca = M_vca.to(device)
        self.M_true = M_true
        self.A_true = A_true
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 损失函数
        self.log_sad_loss = LogSADLoss()
        self.sad_loss = SADLoss()
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'ee_loss': [],
            'abundance_loss': [],
            'val_sad': [],
            'val_rmse': []
        }
        
    def train_ee_network(self, train_loader, epochs=100, lr=1e-4):
        """
        阶段1：训练端元提取网络
        """
        print("\n" + "="*50)
        print("Stage 1: Training Endmember Extraction Network")
        print("="*50)
        
        # 只优化EE网络参数，使用更大的学习率
        optimizer = optim.Adam(self.model.ee_network.parameters(), lr=lr)
        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_loss = float('inf')
        patience = 15  # 增加耐心值
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.ee_network.train()
            epoch_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, x in enumerate(pbar):
                x = x.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                _, Y_hat, M_hat = self.model(x, self.M_vca, return_endmembers=True)
                
                # 计算log(1+SAD)损失
                loss = self.log_sad_loss(Y_hat, x)
                
                # 检查损失是否为有效值
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: Invalid loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.ee_network.parameters(), max_norm=1.0)
                
                # 约束散射模型参数
                self.model.ee_network.dispersion_model.clamp_parameters()
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({'loss': loss.item()})
            
            if batch_count == 0:
                print("Warning: No valid batches in this epoch!")
                continue
                
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}")
            
            # 更新学习率
            scheduler.step(avg_loss)
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.ee_network.state_dict(), 
                          os.path.join(self.save_dir, 'best_ee_network.pth'))
                print(f"  → New best model saved (loss: {best_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        self.model.ee_network.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'best_ee_network.pth'))
        )
        print("EE Network training completed!")
        
    def train_abundance_network(self, train_loader, epochs=100, lr=1e-4):
        """
        阶段2：固定EE网络，训练丰度估计网络
        """
        print("\n" + "="*50)
        print("Stage 2: Training Abundance Estimation Network")
        print("="*50)
        
        # 冻结EE网络
        for param in self.model.ee_network.parameters():
            param.requires_grad = False
        
        # 只优化丰度网络参数
        optimizer = optim.Adam(self.model.abundance_network.parameters(), lr=lr)
        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_loss = float('inf')
        patience = 15  # 增加耐心值
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.abundance_network.train()
            epoch_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, x in enumerate(pbar):
                x = x.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                A, Y_hat = self.model(x, self.M_vca)
                
                # 计算SAD损失
                loss = self.sad_loss(Y_hat, x)
                
                # 检查损失是否为有效值
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: Invalid loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.abundance_network.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({'loss': loss.item()})
            
            if batch_count == 0:
                print("Warning: No valid batches in this epoch!")
                continue
                
            avg_loss = epoch_loss / batch_count
            
            # 评估
            val_sad, val_rmse = self.evaluate(train_loader)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, mSAD={val_sad:.4f}, mRMSE={val_rmse:.4f}")
            
            # 更新学习率
            scheduler.step(avg_loss)
            
            # 记录历史
            self.train_history['epoch'].append(epoch+1)
            self.train_history['abundance_loss'].append(avg_loss)
            self.train_history['val_sad'].append(val_sad)
            self.train_history['val_rmse'].append(val_rmse)
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 
                          os.path.join(self.save_dir, 'best_model.pth'))
                print(f"  → New best model saved (loss: {best_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'best_model.pth'))
        )
        print("Abundance Network training completed!")
        
    def evaluate(self, data_loader):
        """
        评估模型性能
        Returns:
            val_sad: 平均SAD（如果有真实端元）
            val_rmse: 平均RMSE（如果有真实丰度）
        """
        self.model.eval()
        
        all_A_pred = []
        
        with torch.no_grad():
            for x in data_loader:
                x = x.to(self.device)
                A, Y_hat, M_hat = self.model(x, self.M_vca, return_endmembers=True)
                all_A_pred.append(A.cpu())
        
        A_pred = torch.cat(all_A_pred, dim=0)
        
        # 计算SAD（如果有真实端元）
        val_sad = 0.0
        if self.M_true is not None:
            M_hat_final = M_hat.cpu()
            M_true = self.M_true if isinstance(self.M_true, torch.Tensor) else torch.from_numpy(self.M_true).float()
            _, val_sad = compute_endmember_sad(M_hat_final, M_true)
        
        # 计算RMSE（如果有真实丰度）
        val_rmse = 0.0
        if self.A_true is not None:
            _, val_rmse = compute_abundance_rmse(A_pred, self.A_true)
        
        return val_sad, val_rmse
    
    def save_training_log(self):
        """保存训练日志为CSV"""
        log_path = os.path.join(self.save_dir, 'training_log.csv')
        
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Abundance_Loss', 'Val_mSAD', 'Val_mRMSE'])
            
            for i in range(len(self.train_history['epoch'])):
                writer.writerow([
                    self.train_history['epoch'][i],
                    self.train_history['abundance_loss'][i],
                    self.train_history['val_sad'][i],
                    self.train_history['val_rmse'][i]
                ])
        
        print(f"Training log saved to {log_path}")


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    from data_preprocess import JasperRidgePreprocessor
    
    preprocessor = JasperRidgePreprocessor()
    hsi_2d, M_true, A_true = preprocessor.load_and_preprocess()
    
    # VCA提取初始端元
    vca = VCA(n_endmembers=4)
    M_vca = vca.fit(hsi_2d.T)
    M_vca_torch = torch.from_numpy(M_vca).float()
    
    # 创建数据集和数据加载器
    dataset = JasperRidgeDataset(hsi_2d)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 创建模型
    model = DMTSNet(n_bands=198, n_endmembers=4, K=3)
    
    # 创建训练器
    trainer = DMTSTrainer(
        model=model,
        M_vca=M_vca_torch,
        M_true=M_true,
        A_true=A_true,
        device=device,
        save_dir='./results'
    )
    
    # 阶段1：训练EE网络
    trainer.train_ee_network(train_loader, epochs=100, lr=1e-4)
    
    # 阶段2：训练丰度网络
    trainer.train_abundance_network(train_loader, epochs=100, lr=1e-4)
    
    # 保存训练日志
    trainer.save_training_log()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
