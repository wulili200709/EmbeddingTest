"""
Embedding 特征可视化示例
从训练好的模型中提取特征并可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import qr_core
import cv2
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_embeddings(model_path, ok_images, ng_images):
    """
    可视化OK和NG样本的特征分布
    
    Args:
        model_path: 模型文件路径
        ok_images: OK图片路径列表
        ng_images: NG图片路径列表
    """
    print("加载模型...")
    model = qr_core.load_register_model_npz(model_path)
    
    # 重新加载backbone（因为模型只保存了名称，不保存torch模型）
    print(f"加载 {model.backbone} backbone...")
    feat_net, out_ch = qr_core.load_backbone(model.backbone, device=model.device)
    
    print(f"特征维度: {out_ch}")
    
    # 提取所有样本的特征
    print(f"提取OK样本特征（{len(ok_images)}张）...")
    ok_features = []
    for i, img_path in enumerate(ok_images, 1):
        print(f"  OK {i}/{len(ok_images)}: {os.path.basename(img_path)}")
        # 直接使用已保存的embedding（模型训练时已经提取过）
        # 但为了可视化，我们可以重新提取以确保一致性
        try:
            feat = qr_core.embed_one(img_path, feat_net, label_name=model.label_name, device=model.device)
            ok_features.append(feat)
        except Exception as e:
            print(f"    跳过（错误: {e}）")
            continue
    
    print(f"提取NG样本特征（{len(ng_images)}张）...")
    ng_features = []
    for i, img_path in enumerate(ng_images, 1):
        print(f"  NG {i}/{len(ng_images)}: {os.path.basename(img_path)}")
        try:
            feat = qr_core.embed_one(img_path, feat_net, label_name=model.label_name, device=model.device)
            ng_features.append(feat)
        except Exception as e:
            print(f"    跳过（错误: {e}）")
            continue
    
    if not ok_features or not ng_features:
        print("错误：没有足够的样本")
        print(f"OK特征: {len(ok_features)}, NG特征: {len(ng_features)}")
        return
    
    # 合并所有特征
    all_features = np.vstack([
        np.array(ok_features),
        np.array(ng_features)
    ])
    
    # 创建标签
    labels = ['OK'] * len(ok_features) + ['NG'] * len(ng_features)
    
    print(f"总共 {len(all_features)} 个样本，{all_features.shape[1]} 维特征")
    print(f"OK: {len(ok_features)} 个, NG: {len(ng_features)} 个")
    
    # 使用t-SNE降维到2D
    print("使用t-SNE降维到2D...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
    features_2d = tsne.fit_transform(all_features)
    
    # 分离OK和NG的2D坐标
    ok_2d = features_2d[:len(ok_features)]
    ng_2d = features_2d[len(ok_features):]
    
    # 画图
    plt.figure(figsize=(10, 8))
    plt.scatter(ok_2d[:, 0], ok_2d[:, 1], c='blue', label='OK样本', s=100, alpha=0.6, edgecolors='black')
    plt.scatter(ng_2d[:, 0], ng_2d[:, 1], c='red', label='NG样本', s=100, alpha=0.6, edgecolors='black')
    
    # 标记原型（中心点）
    ok_center = ok_2d.mean(axis=0)
    ng_center = ng_2d.mean(axis=0)
    plt.scatter([ok_center[0]], [ok_center[1]], c='darkblue', marker='*', s=500, 
                label='OK原型', edgecolors='black', linewidths=2)
    plt.scatter([ng_center[0]], [ng_center[1]], c='darkred', marker='*', s=500, 
                label='NG原型', edgecolors='black', linewidths=2)
    
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    plt.title(f'特征向量可视化 ({all_features.shape[1]}维 → 2维)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    output_path = 'embedding_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化图片已保存到: {output_path}")
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 请修改为你的实际路径
    MODEL_PATH = ".qr_session/Screw/register_model_efficientnet_b0.npz"
    
    # 从session.json读取文件列表
    import json
    session_file = ".qr_session/Screw/session.json"
    
    if os.path.exists(session_file):
        with open(session_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            ok_images = data.get('ok_files', [])
            ng_images = data.get('ng_files', [])
    else:
        print("找不到session.json，请手动指定图片路径")
        ok_images = []
        ng_images = []
    
    if os.path.exists(MODEL_PATH) and ok_images and ng_images:
        visualize_embeddings(MODEL_PATH, ok_images, ng_images)
    else:
        print("错误：模型文件不存在或没有训练样本")
        print(f"模型路径: {MODEL_PATH}")
        print(f"OK样本数: {len(ok_images)}")
        print(f"NG样本数: {len(ng_images)}")
