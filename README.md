# DMTS-Net: Blind Unmixing Using Dispersion Model-Based Autoencoder

## 项目简介

本项目复现了论文《Blind Unmixing Using Dispersion Model-Based Autoencoder to Address Spectral Variability》中的DMTS-Net模型，并在Jasper Ridge高光谱数据集上进行验证。

### 核心特性

- **双流网络架构**：端元提取网络(EE Network) + 丰度估计网络(Abundance Network)
- **物理驱动模型**：基于散射模型(Dispersion Model)处理光谱变异
- **分阶段训练**：先训练EE网络，再固定权重训练丰度网络
- **完整评估**：SAD/RMSE指标 + 端元/丰度可视化

## 环境配置

### 1. 创建虚拟环境（推荐）

```bash
# 使用conda
conda create -n dmts python=3.8
conda activate dmts

# 或使用venv
python -m venv dmts_env
source dmts_env/bin/activate  # Linux/Mac
# dmts_env\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖列表

- PyTorch >= 1.10.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0

## 快速开始

### 完整流程（训练+评估）

```bash
python main.py --mode both
```

### 仅训练

```bash
python main.py --mode train --ee_epochs 100 --abundance_epochs 100
```

### 仅评估

```bash
python main.py --mode eval
```

### 自定义参数

```bash
python main.py \
    --n_endmembers 4 \
    --K 3 \
    --batch_size 32 \
    --lr 1e-4 \
    --ee_epochs 100 \
    --abundance_epochs 100 \
    --save_dir ./my_results
```

## 项目结构

```
Autoencoder/
├── data/                          # 数据目录
│   ├── jasperRidge2_R198.mat     # Jasper Ridge数据集
│   └── preprocess_report.txt     # 预处理报告
├── results/                       # 结果目录
│   ├── best_model.pth            # 最佳完整模型
│   ├── best_ee_network.pth       # 最佳EE网络
│   ├── training_log.csv          # 训练日志
│   ├── performance_metrics.txt   # 性能指标
│   ├── endmembers_comparison.png # 端元对比图
│   ├── abundance_maps.png        # 丰度分布图
│   ├── abundance_comparison.png  # 丰度对比图
│   └── endmember_matching.txt    # 端元匹配报告（检查工具输出）
├── data_preprocess.py            # 数据预处理
├── vca.py                        # VCA算法
├── dispersion_model.py           # 散射模型
├── dmts_net.py                   # DMTS-Net网络
├── loss_functions.py             # 损失函数
├── train.py                      # 训练脚本
├── evaluate.py                   # 评估脚本
├── check_endmember_order.py      # 端元顺序检查工具
├── main.py                       # 主函数
├── requirements.txt              # 依赖配置
└── README.md                     # 项目说明
```

## 模块说明

### 1. 数据预处理 (`data_preprocess.py`)

- 自动下载Jasper Ridge数据集
- 去除水汽/大气影响波段（224→198波段）
- 裁剪为100×100像素
- 归一化到[0,1]
- 转换为2D格式(N×L)

### 2. VCA算法 (`vca.py`)

- 提取初始端元M_vca
- 基于最大体积单纯形
- 输出4个端元（Tree, Water, Soil, Road）

### 3. 散射模型 (`dispersion_model.py`)

- 物理参数：ρ（密度）、ω₀（共振频率）、γ（阻尼）、εᵣ（介电常数）
- 基于质量弹簧方程（K=3）
- 生成变异端元M_var

### 4. DMTS-Net网络 (`dmts_net.py`)

**端元提取网络 (EE Network)**
- 4层全连接 + LeakyReLU(α=0.4) + BatchNorm
- 集成散射模型
- 输出：M_hat = M_var + M_vca

**丰度估计网络 (Abundance Network)**
- 编码器：5层1D卷积 + 2层FC (Tanh激活)
- 解码器：1层FC（无激活/偏置）
- 丰度约束：非负 + 和为1

### 5. 损失函数 (`loss_functions.py`)

- **EE网络损失**：L₁ = (1/n)Σlog(SAD(yᵢ, ŷᵢ))
- **丰度网络损失**：L₂ = (1/n)ΣSAD(yᵢ, ŷᵢ)
- **评估指标**：SAD（端元）、RMSE（丰度）

### 6. 训练策略 (`train.py`)

1. **阶段1**：训练EE网络（100 epochs）
   - 优化散射模型参数
   - log(SAD)损失
   - 早停机制（patience=10）

2. **阶段2**：固定EE网络，训练丰度网络（100 epochs）
   - 仅优化丰度网络参数
   - SAD损失
   - 早停机制（patience=10）

### 7. 评估脚本 (`evaluate.py`)

- 提取预测端元和丰度
- **自动端元匹配**：使用SAD最小化进行端元顺序匹配
- 计算SAD/RMSE指标
- 生成可视化图（300 DPI）
  - 端元光谱对比图
  - 丰度空间分布图
  - 真实vs估计对比图

### 8. 端元顺序检查 (`check_endmember_order.py`)

- 分析VCA端元与真实端元的对应关系
- 计算相关系数矩阵
- 贪心算法寻找最佳匹配

## 性能基准

论文在Jasper Ridge数据集上的DMTS-Net最优结果：

| 指标 | 论文值 |
|------|--------|
| 平均SAD (mSAD) | 0.0278 |
| 平均RMSE (mRMSE) | 0.0568 |

## 命令行参数

```
--data_dir          数据保存目录（默认：./data）
--save_dir          结果保存目录（默认：./results）
--n_bands           波段数（默认：198）
--n_endmembers      端元数（默认：4）
--K                 质量弹簧方程数（默认：3）
--batch_size        批次大小（默认：32）
--ee_epochs         EE网络训练轮数（默认：100）
--abundance_epochs  丰度网络训练轮数（默认：100）
--lr                学习率（默认：1e-4）
--mode              运行模式：train/eval/both（默认：both）
--seed              随机种子（默认：42）
--cuda/--no-cuda    是否使用GPU（默认：自动检测）
```

## 常见问题

### 1. CUDA内存不足

降低批次大小：
```bash
python main.py --batch_size 16
```

### 2. 数据下载失败

手动下载Jasper Ridge数据集：
- 下载地址：http://www.ehu.eus/ccwintco/uploads/2/22/Jasper.mat
- 保存到：`./data/jasperRidge2_R198.mat`

### 3. 训练时间过长

减少训练轮数：
```bash
python main.py --ee_epochs 50 --abundance_epochs 50
```

## 引用

如果使用本代码，请引用原论文：

```
@article{dmts2023,
  title={Blind Unmixing Using Dispersion Model-Based Autoencoder to Address Spectral Variability},
  author={...},
  journal={...},
  year={2023}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue。
