import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_all_components(root_path, freq, series_idx=0, save_dir='vis_results_full'):
    """
    可视化 M4 数据集的全部 EMD 分量 (不合并)
    args:
        root_path: 数据集根目录
        freq: 频率 ('Hourly', 'Daily', etc.)
        series_idx: 序列索引
    """
    # 1. 构建文件路径
    train_file = os.path.join(root_path, f"M4_{freq}_train_cd.npy")
    test_file = os.path.join(root_path, f"M4_{freq}_test_cd.npy")
    
    if not os.path.exists(train_file):
        print(f"Error: File not found {train_file}")
        return

    print(f"Loading {freq} data...")
    # 加载 Object Array
    train_list = np.load(train_file, allow_pickle=True)
    test_list = np.load(test_file, allow_pickle=True) if os.path.exists(test_file) else None
    
    # 检查索引
    if series_idx >= len(train_list):
        print(f"Error: Index {series_idx} out of bounds. Max index is {len(train_list)-1}")
        return
        
    # 获取单条序列数据 [T, N_IMFS]
    imfs_train = train_list[series_idx]
    
    # 获取分量数量
    T_train, n_imfs = imfs_train.shape
    
    # 准备 Test 数据
    has_test = test_list is not None and series_idx < len(test_list)
    if has_test:
        imfs_test = test_list[series_idx]
        T_test = imfs_test.shape[0]
        # 时间轴
        t_train = np.arange(T_train)
        t_test = np.arange(T_train, T_train + T_test)
    else:
        t_train = np.arange(T_train)
        t_test = None

    # 计算原始信号 (Sum of all IMFs)
    orig_train = np.sum(imfs_train, axis=-1)
    orig_test = np.sum(imfs_test, axis=-1) if has_test else None

    # ==========================================
    # 开始绘图
    # ==========================================
    # 动态计算高度：每个分量给 2 inch 高度，外加原始信号
    fig_height = 2.5 * (n_imfs + 1)
    fig, axs = plt.subplots(n_imfs + 1, 1, figsize=(14, fig_height), sharex=True)
    
    plt.suptitle(f"M4 {freq} - Series ID: {series_idx} (Full {n_imfs} IMFs)", fontsize=16, y=0.99)
    
    # --- 1. 绘制原始信号 ---
    axs[0].plot(t_train, orig_train, color='black', linewidth=1.5, label='Train (History)')
    if has_test:
        axs[0].plot(t_test, orig_test, color='red', linewidth=1.5, label='Test (Ground Truth)')
    axs[0].set_title("Original Signal (Reconstructed)", fontsize=12, fontweight='bold')
    axs[0].set_ylabel("Amplitude")
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)

    # --- 2. 绘制每个 IMF 分量 ---
    for i in range(n_imfs):
        ax = axs[i + 1]
        
        # 绘制 Train 部分
        ax.plot(t_train, imfs_train[:, i], color='#1f77b4', linewidth=1.0, alpha=0.8, label='Train IMF')
        
        # 绘制 Test 部分
        if has_test:
            # 这里的 Test IMF 是基于全局分解截取的，代表未来的真实分量趋势
            ax.plot(t_test, imfs_test[:, i], color='#ff7f0e', linewidth=1.0, label='Test IMF')
            
        # 标签和美化
        ax.set_ylabel(f"IMF {i}", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 右侧标注分量特性 (High/Low Freq)
        if i == 0:
            ax.text(1.01, 0.5, "High Freq\n(Noise)", transform=ax.transAxes, va='center', ha='left', color='gray')
        elif i == n_imfs - 1:
            ax.text(1.01, 0.5, "Low Freq\n(Trend)", transform=ax.transAxes, va='center', ha='left', color='gray')

    axs[-1].set_xlabel("Time Step", fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"FullVis_{freq}_{series_idx}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved full visualization to {save_path}")
    # plt.show() # 如果在服务器运行，请注释掉此行

if __name__ == "__main__":
    # 配置路径
    ROOT_PATH = "./dataset/m4" 
    
    # 想要查看的配置
    FREQUENCY = "Monthly"   # 试试 'Daily' 或 'Hourly'
    SERIES_IDX = 300      # 更改索引查看不同序列
    
    visualize_all_components(ROOT_PATH, FREQUENCY, SERIES_IDX)