@echo off
REM filepath: c:\Users\14704\VSCodeProjects\RemoteSensing\Autoencoder\run_experiment.bat
REM DMTS-Net实验运行脚本

echo ==========================================
echo DMTS-Net Experiment Runner
echo ==========================================

REM 激活虚拟环境（如果使用）
REM call dmts_env\Scripts\activate

REM 创建结果目录
if not exist "results" mkdir results
if not exist "data" mkdir data

REM 运行完整流程
echo.
echo Running full pipeline (training + evaluation)...
python main.py ^
    --mode both ^
    --n_endmembers 4 ^
    --K 3 ^
    --batch_size 32 ^
    --lr 1e-3 ^
    --ee_epochs 50 ^
    --abundance_epochs 50 ^
    --save_dir ./results ^
    --cuda

echo.
echo ==========================================
echo Experiment completed!
echo Results saved to ./results/
echo ==========================================

pause