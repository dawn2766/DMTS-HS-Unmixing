"""
DMTS-Net主函数
整合数据预处理、模型训练、评估的完整流程
"""

import torch
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader

from data_preprocess import JasperRidgePreprocessor, JasperRidgeDataset
from vca import VCA
from dmts_net import DMTSNet
from train import DMTSTrainer
from evaluate import DMTSEvaluator


def main(args):
    """主函数"""
    
    print("="*70)
    print("DMTS-Net: Blind Unmixing Using Dispersion Model-Based Autoencoder")
    print("="*70)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ========== 阶段1: 数据预处理 ==========
    print("\n" + "="*70)
    print("Stage 1: Data Preprocessing")
    print("="*70)
    
    preprocessor = JasperRidgePreprocessor(data_path=args.data_dir)
    hsi_2d, M_true, A_true = preprocessor.load_and_preprocess()
    
    # 创建数据集
    dataset = JasperRidgeDataset(hsi_2d)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    eval_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    print(f"\nDataset size: {len(dataset)} pixels")
    print(f"Batch size: {args.batch_size}")
    
    # ========== 阶段2: VCA初始端元提取 ==========
    print("\n" + "="*70)
    print("Stage 2: VCA Endmember Extraction")
    print("="*70)
    
    vca = VCA(n_endmembers=args.n_endmembers)
    M_vca = vca.fit(hsi_2d.T)
    M_vca_torch = torch.from_numpy(M_vca).float()
    
    print(f"Extracted {args.n_endmembers} initial endmembers")
    
    # ========== 阶段3: 模型创建 ==========
    print("\n" + "="*70)
    print("Stage 3: Model Creation")
    print("="*70)
    
    model = DMTSNet(
        n_bands=args.n_bands, 
        n_endmembers=args.n_endmembers, 
        K=args.K
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: DMTS-Net")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of endmembers (P): {args.n_endmembers}")
    print(f"Number of bands (L): {args.n_bands}")
    print(f"Mass-spring oscillators (K): {args.K}")
    
    # ========== 阶段4: 模型训练 ==========
    if args.mode in ['train', 'both']:
        print("\n" + "="*70)
        print("Stage 4: Model Training")
        print("="*70)
        
        trainer = DMTSTrainer(
            model=model,
            M_vca=M_vca_torch,
            M_true=M_true,
            A_true=A_true,
            device=device,
            save_dir=args.save_dir
        )
        
        # 阶段1：训练EE网络
        trainer.train_ee_network(
            train_loader, 
            epochs=args.ee_epochs, 
            lr=args.lr
        )
        
        # 阶段2：训练丰度网络
        trainer.train_abundance_network(
            train_loader, 
            epochs=args.abundance_epochs, 
            lr=args.lr
        )
        
        # 保存训练日志
        trainer.save_training_log()
        
        print("\nTraining completed!")
        print(f"Models saved to {args.save_dir}")
    
    # ========== 阶段5: 模型评估 ==========
    if args.mode in ['eval', 'both']:
        print("\n" + "="*70)
        print("Stage 5: Model Evaluation")
        print("="*70)
        
        # 加载最佳模型
        model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")
            if args.mode == 'eval':
                print("Please train the model first using --mode train")
                return
        
        evaluator = DMTSEvaluator(
            model=model,
            M_vca=M_vca_torch,
            M_true=M_true,
            A_true=A_true,
            device=device,
            save_dir=args.save_dir
        )
        
        results = evaluator.run_full_evaluation(eval_loader)
        
        print("\nEvaluation completed!")
        print(f"Results saved to {args.save_dir}")
    
    print("\n" + "="*70)
    print("All stages completed successfully!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMTS-Net for Jasper Ridge Dataset')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据保存目录')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='结果保存目录')
    
    # 模型参数
    parser.add_argument('--n_bands', type=int, default=198,
                       help='波段数')
    parser.add_argument('--n_endmembers', type=int, default=4,
                       help='端元数（Jasper Ridge=4）')
    parser.add_argument('--K', type=int, default=3,
                       help='质量弹簧方程数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--ee_epochs', type=int, default=100,
                       help='EE网络训练轮数')
    parser.add_argument('--abundance_epochs', type=int, default=100,
                       help='丰度网络训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,  # 从1e-4改为1e-3
                       help='学习率')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='数据加载线程数')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='both',
                       choices=['train', 'eval', 'both'],
                       help='运行模式: train=仅训练, eval=仅评估, both=训练+评估')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='使用CUDA（如果可用）')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                       help='不使用CUDA')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(args)
