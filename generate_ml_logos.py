"""
生成机器学习术语Logo（图标+文字）
类似 Snowflake 风格的专业Logo设计生成脚本
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon
import numpy as np

# 配置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLOR_BLUE = '#1E40AF'
COLOR_PURPLE = '#7C3AED'
COLOR_TEXT = '#1F2937'

def create_logo(title, subtitle, icon_func, filename):
    """
    创建Logo
    
    Args:
        title: 主标题
        subtitle: 副标题（可选）
        icon_func: 绘制图标的函数
        filename: 保存文件名
    """
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    # 绘制图标（左侧）
    icon_ax = fig.add_axes([0.05, 0.2, 0.2, 0.6])
    icon_ax.set_xlim(0, 1)
    icon_ax.set_ylim(0, 1)
    icon_ax.axis('off')
    icon_func(icon_ax)
    
    # 绘制文字（右侧）
    if subtitle:
        ax.text(3.5, 1.2, title, fontsize=32, fontweight='bold', 
                color=COLOR_TEXT, va='center', ha='left')
        ax.text(3.5, 0.65, subtitle, fontsize=20, fontweight='300', 
                color=COLOR_BLUE, va='center', ha='left')
    else:
        ax.text(3.5, 1.0, title, fontsize=32, fontweight='bold', 
                color=COLOR_TEXT, va='center', ha='left')
    
    # 保存
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ 已生成: {filename}")
    plt.close()


# ===== 图标绘制函数 =====

def icon_feature_extraction(ax):
    """特征提取引擎图标 - 漏斗+神经网络"""
    # 漏斗形状
    funnel = Polygon([(0.3, 0.8), (0.7, 0.8), (0.6, 0.4), (0.4, 0.4)],
                     facecolor=COLOR_BLUE, edgecolor='none', alpha=0.7)
    ax.add_patch(funnel)
    
    # 输出节点
    for i, y in enumerate([0.3, 0.2, 0.1]):
        circle = Circle((0.5, y), 0.04, facecolor=COLOR_PURPLE, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
    
    # 电路线条
    ax.plot([0.5, 0.5], [0.4, 0.34], color=COLOR_PURPLE, linewidth=2)


def icon_deep_features(ax):
    """深度特征图标 - 多层神经网络"""
    layers = [0.8, 0.6, 0.4, 0.2]
    for i, y in enumerate(layers):
        alpha = 0.4 + (i * 0.15)  # 0.4, 0.55, 0.70, 0.85 - 确保不超过1.0
        rect = Rectangle((0.2, y-0.05), 0.6, 0.08, 
                        facecolor=COLOR_BLUE, edgecolor='none', alpha=alpha)
        ax.add_patch(rect)
        
        # 节点
        for x in [0.3, 0.5, 0.7]:
            circle_alpha = min(0.9, alpha + 0.2)  # 确保不超过1.0
            circle = Circle((x, y), 0.03, facecolor=COLOR_PURPLE, 
                          edgecolor='white', linewidth=1, alpha=circle_alpha)
            ax.add_patch(circle)


def icon_feature_vector(ax):
    """特征向量图标 - 箭头+坐标"""
    # 坐标轴
    ax.plot([0.2, 0.8], [0.3, 0.3], color=COLOR_BLUE, linewidth=2, alpha=0.3)
    ax.plot([0.2, 0.2], [0.3, 0.8], color=COLOR_BLUE, linewidth=2, alpha=0.3)
    
    # 向量箭头
    arrow = FancyArrowPatch((0.25, 0.35), (0.75, 0.75),
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=3, color=COLOR_PURPLE)
    ax.add_patch(arrow)
    
    # 终点标记
    circle = Circle((0.75, 0.75), 0.05, facecolor=COLOR_PURPLE, 
                   edgecolor='white', linewidth=2)
    ax.add_patch(circle)


def icon_1280d_embedding(ax):
    """高维嵌入图标 - 3D立方体+点云"""
    # 立方体框架
    cube_lines = [
        ([0.3, 0.6], [0.3, 0.3]),  # 底部前边
        ([0.6, 0.6], [0.3, 0.6]),  # 底部右边
        ([0.3, 0.5], [0.6, 0.8]),  # 左竖边
        ([0.6, 0.8], [0.6, 0.8]),  # 右竖边
        ([0.5, 0.8], [0.8, 0.8]),  # 顶部
    ]
    for x, y in cube_lines:
        ax.plot(x, y, color=COLOR_BLUE, linewidth=2, alpha=0.5)
    
    # 内部点云
    np.random.seed(42)
    for _ in range(15):
        x = np.random.uniform(0.35, 0.7)
        y = np.random.uniform(0.35, 0.75)
        circle = Circle((x, y), 0.02, facecolor=COLOR_PURPLE, alpha=0.6)
        ax.add_patch(circle)


def icon_prototypical_learning(ax):
    """原型学习图标 - 中心原型+周围样本"""
    # 中心原型（星形）
    star = Circle((0.5, 0.5), 0.08, facecolor=COLOR_PURPLE, 
                 edgecolor='white', linewidth=2)
    ax.add_patch(star)
    
    # 周围样本点
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in angles:
        x = 0.5 + 0.25 * np.cos(angle)
        y = 0.5 + 0.25 * np.sin(angle)
        circle = Circle((x, y), 0.04, facecolor=COLOR_BLUE, 
                       edgecolor='white', linewidth=1, alpha=0.7)
        ax.add_patch(circle)
        # 连接线
        ax.plot([0.5, x], [0.5, y], color=COLOR_BLUE, 
               linewidth=1, alpha=0.3, linestyle='--')


def icon_metric_learning(ax):
    """度量学习图标 - 距离测量"""
    # 两个点
    circle1 = Circle((0.3, 0.5), 0.06, facecolor=COLOR_BLUE, 
                    edgecolor='white', linewidth=2)
    circle2 = Circle((0.7, 0.5), 0.06, facecolor=COLOR_PURPLE, 
                    edgecolor='white', linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # 距离线（双向箭头）
    arrow = FancyArrowPatch((0.36, 0.5), (0.64, 0.5),
                           arrowstyle='<->', mutation_scale=20, 
                           linewidth=2, color=COLOR_TEXT)
    ax.add_patch(arrow)
    
    # 距离标记
    ax.text(0.5, 0.62, 'd', fontsize=14, ha='center', 
           fontweight='bold', color=COLOR_TEXT)


def icon_feature_space(ax):
    """特征空间图标 - 坐标系+散点"""
    # 坐标轴
    ax.plot([0.2, 0.8], [0.2, 0.2], color=COLOR_BLUE, linewidth=2)
    ax.plot([0.2, 0.2], [0.2, 0.8], color=COLOR_BLUE, linewidth=2)
    
    # 箭头
    ax.annotate('', xy=(0.82, 0.2), xytext=(0.78, 0.2),
               arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2))
    ax.annotate('', xy=(0.2, 0.82), xytext=(0.2, 0.78),
               arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2))
    
    # 散点
    np.random.seed(42)
    for _ in range(12):
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(0.3, 0.7)
        circle = Circle((x, y), 0.025, facecolor=COLOR_PURPLE, alpha=0.6)
        ax.add_patch(circle)


def icon_dimensionality_reduction(ax):
    """降维可视化图标 - 3D→2D"""
    # 3D立方体（左侧）
    cube = Rectangle((0.15, 0.4), 0.2, 0.2, facecolor=COLOR_BLUE, 
                    edgecolor='none', alpha=0.3)
    ax.add_patch(cube)
    ax.plot([0.15, 0.25], [0.6, 0.75], color=COLOR_BLUE, linewidth=1)
    ax.plot([0.35, 0.45], [0.6, 0.75], color=COLOR_BLUE, linewidth=1)
    
    # 转换箭头
    arrow = FancyArrowPatch((0.47, 0.5), (0.58, 0.5),
                           arrowstyle='->', mutation_scale=25, 
                           linewidth=3, color=COLOR_PURPLE)
    ax.add_patch(arrow)
    
    # 2D平面（右侧）
    plane = Rectangle((0.62, 0.35), 0.25, 0.3, facecolor=COLOR_PURPLE, 
                     edgecolor='none', alpha=0.5)
    ax.add_patch(plane)


def icon_tsne(ax):
    """t-SNE降维图标 - 投影映射"""
    # 高维点云（上方）
    np.random.seed(42)
    for _ in range(8):
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(0.65, 0.85)
        circle = Circle((x, y), 0.02, facecolor=COLOR_BLUE, alpha=0.5)
        ax.add_patch(circle)
    
    # 投影箭头
    for i in range(3):
        x1 = 0.3 + i * 0.2
        ax.plot([x1, x1 + 0.05], [0.65, 0.45], color=COLOR_PURPLE, 
               linewidth=1, alpha=0.4, linestyle='--')
    
    # 2D散点图（下方）
    ax.plot([0.25, 0.75], [0.25, 0.25], color=COLOR_BLUE, linewidth=1.5)
    ax.plot([0.25, 0.25], [0.25, 0.45], color=COLOR_BLUE, linewidth=1.5)
    
    for _ in range(8):
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(0.28, 0.42)
        circle = Circle((x, y), 0.025, facecolor=COLOR_PURPLE, alpha=0.7)
        ax.add_patch(circle)


def icon_clustering(ax):
    """聚类分析图标 - 分组点云"""
    # 三个聚类
    clusters = [
        (0.3, 0.7, COLOR_BLUE),
        (0.7, 0.7, COLOR_PURPLE),
        (0.5, 0.3, '#10B981')  # 绿色
    ]
    
    np.random.seed(42)
    for cx, cy, color in clusters:
        # 聚类包围圈
        circle_bg = Circle((cx, cy), 0.15, facecolor=color, alpha=0.15)
        ax.add_patch(circle_bg)
        
        # 聚类中的点
        for _ in range(5):
            x = cx + np.random.uniform(-0.08, 0.08)
            y = cy + np.random.uniform(-0.08, 0.08)
            point = Circle((x, y), 0.02, facecolor=color, alpha=0.8)
            ax.add_patch(point)


# ===== 主程序 =====

if __name__ == "__main__":
    print("开始生成机器学习术语Logo...\n")
    
    logos = [
        ("Feature Extraction", "Engine", icon_feature_extraction, "logo_feature_extraction.png"),
        ("Deep Features", "", icon_deep_features, "logo_deep_features.png"),
        ("Feature Vector", "", icon_feature_vector, "logo_feature_vector.png"),
        ("1280-D Feature", "Embedding", icon_1280d_embedding, "logo_1280d_embedding.png"),
        ("Prototypical", "Learning", icon_prototypical_learning, "logo_prototypical_learning.png"),
        ("Metric Learning", "", icon_metric_learning, "logo_metric_learning.png"),
        ("Feature Space", "", icon_feature_space, "logo_feature_space.png"),
        ("Dimensionality", "Reduction", icon_dimensionality_reduction, "logo_dimensionality_reduction.png"),
        ("t-SNE Projection", "", icon_tsne, "logo_tsne_projection.png"),
        ("Clustering", "Analysis", icon_clustering, "logo_clustering_analysis.png"),
    ]
    
    for title, subtitle, icon_func, filename in logos:
        create_logo(title, subtitle, icon_func, filename)
    
    print(f"\n✅ 完成！共生成 {len(logos)} 个Logo")
    print("\n所有Logo已保存为高分辨率PNG文件（300 DPI）")
    print("可直接用于PPT演示！")
