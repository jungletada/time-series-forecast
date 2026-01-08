import os
import numpy as np
import matplotlib.pyplot as plt

def get_top_k_flow_nodes(data_path, k=3):
    """
    加载 PEMS 数据并返回平均流量最大的 k 个传感器的索引和数据
    """
    try:
        # 1. 加载数据
        # PEMS .npz 结构通常是 (T, N, F) 或 (T, N)
        raw_data = np.load(data_path)['data']
        print(f"原始数据形状: {raw_data.shape}")
        
        # 2. 提取车流量通道 (Flow)
        # 假设第 0 个通道是 Flow (大部分 PEMS 标准)
        if raw_data.ndim == 3:
            flow_data = raw_data[:, :, 0]
        else:
            flow_data = raw_data # 如果已经是 2D，直接用
            
        # 3. 计算每个传感器的平均流量 (按时间轴 axis=0 取平均)
        # 这代表了该路段的繁忙程度
        mean_flows = np.mean(flow_data, axis=0)
        
        # 4. 获取 Top-K 索引
        # argsort 返回从小到大的索引，[-k:] 取最后 k 个，[::-1] 倒序排列变成从大到小
        top_k_indices = np.argsort(mean_flows)[-k:][::-1]
        
        print(f"\n>>> 筛选出的 Top-{k} 流量最大传感器索引: {top_k_indices}")
        for rank, idx in enumerate(top_k_indices):
            print(f"    第 {rank+1} 名 (Node {idx}): 平均流量 = {mean_flows[idx]:.2f}")
            
        # 5. 提取这 k 个传感器的数据
        # 形状变为 (T, k)
        selected_data = flow_data[:, top_k_indices]
        
        return top_k_indices, selected_data
    
    except Exception as e:
        print(f"读取或处理出错: {e}")
        return None, None

if __name__ == '__main__':
    root_path = 'dataset/PEMS/'
    data_path = 'PEMS08.npz'
    data_file = os.path.join(root_path, data_path)
    top_indices, top_data = get_top_k_flow_nodes(data_file, k=3)

    # ================= 可视化检查 =================
    if top_data is not None:
        # 只画前 288*2 个点 (约2天的数据)，避免太密看不清
        plot_len = 288 * 2 
        
        plt.figure(figsize=(12, 6))
        for i in range(top_data.shape[1]):
            node_idx = top_indices[i]
            plt.plot(top_data[:plot_len, i], label=f'Node {node_idx} (Rank {i+1})')
        
        plt.title(f'Top 3 High Traffic Sensors (First 2 Days)')
        plt.xlabel('Time Steps (5 min intervals)')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('top_3_high_traffic_sensors.png')
        
        print("\n=== 模型实验参数建议 ===")
        print(f"--enc_in {len(top_indices)}")
        print(f"--dec_in {len(top_indices)}")
        print(f"--c_out {len(top_indices)}")
        print(f"输入数据形状: {top_data.shape} (Time, Nodes)")