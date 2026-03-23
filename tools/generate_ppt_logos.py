"""
为PPT生成3个专业Logo（Snowflake风格）
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLOR_BLUE = '#1E40AF'
COLOR_PURPLE = '#7C3AED'
COLOR_TEXT = '#1F2937'

def create_logo(title, subtitle, icon_func, filename):
    """创建Logo"""
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    # 图标（左侧）
    icon_ax = fig.add_axes([0.05, 0.2, 0.2, 0.6])
    icon_ax.set_xlim(0, 1)
    icon_ax.set_ylim(0, 1)
    icon_ax.axis('off')
    icon_func(icon_ax)
    
    # 文字（右侧）
    if subtitle:
        ax.text(3.5, 1.2, title, fontsize=32, fontweight='bold', 
                color=COLOR_TEXT, va='center', ha='left')
        ax.text(3.5, 0.65, subtitle, fontsize=20, fontweight='300', 
                color=COLOR_BLUE, va='center', ha='left')
    else:
        ax.text(3.5, 1.0, title, fontsize=32, fontweight='bold', 
                color=COLOR_TEXT, va='center', ha='left')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ 已生成: {filename}")
    plt.close()


def icon_feature_engine(ax):
    """特征引擎图标 - 齿轮+神经网络"""
    # 齿轮外圈
    theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in theta:
        x = 0.5 + 0.25 * np.cos(angle)
        y = 0.5 + 0.25 * np.sin(angle)
        rect = Rectangle((x-0.04, y-0.08), 0.08, 0.16, 
                        facecolor=COLOR_BLUE, edgecolor='none', alpha=0.6)
        ax.add_patch(rect)
    
    # 中心圆
    center = Circle((0.5, 0.5), 0.15, facecolor=COLOR_PURPLE, 
                   edgecolor='white', linewidth=2)
    ax.add_patch(center)
    
    # 中心节点
    for i, (dx, dy) in enumerate([(0, 0), (-0.06, 0), (0.06, 0), (0, -0.06), (0, 0.06)]):
        circle = Circle((0.5+dx, 0.5+dy), 0.02, facecolor='white', alpha=0.8)
        ax.add_patch(circle)


def icon_embedding(ax):
    """嵌入空间图标 - 3D空间+映射"""
    # 3D立方体框架
    cube_lines = [
        ([0.25, 0.55], [0.35, 0.35]),
        ([0.55, 0.55], [0.35, 0.65]),
        ([0.25, 0.45], [0.65, 0.85]),
        ([0.55, 0.75], [0.65, 0.85]),
        ([0.45, 0.75], [0.85, 0.85]),
    ]
    for x, y in cube_lines:
        ax.plot(x, y, color=COLOR_BLUE, linewidth=2.5, alpha=0.6)
    
    # 内部嵌入点
    np.random.seed(42)
    for _ in range(12):
        x = np.random.uniform(0.35, 0.65)
        y = np.random.uniform(0.45, 0.75)
        circle = Circle((x, y), 0.025, facecolor=COLOR_PURPLE, alpha=0.7)
        ax.add_patch(circle)


def icon_feature_visualization(ax):
    """特征可视化图标 - 散点图+可视化"""
    # 坐标轴
    ax.plot([0.2, 0.8], [0.2, 0.2], color=COLOR_BLUE, linewidth=2.5)
    ax.plot([0.2, 0.2], [0.2, 0.8], color=COLOR_BLUE, linewidth=2.5)
    
    # 箭头
    ax.annotate('', xy=(0.82, 0.2), xytext=(0.78, 0.2),
               arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2.5))
    ax.annotate('', xy=(0.2, 0.82), xytext=(0.2, 0.78),
               arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=2.5))
    
    # 两类散点（OK vs NG）
    np.random.seed(42)
    # 蓝色点群（OK）
    for _ in range(8):
        x = np.random.uniform(0.3, 0.5)
        y = np.random.uniform(0.5, 0.7)
        circle = Circle((x, y), 0.03, facecolor=COLOR_BLUE, 
                       edgecolor='white', linewidth=1, alpha=0.7)
        ax.add_patch(circle)
    
    # 紫色点群（NG）
    for _ in range(8):
        x = np.random.uniform(0.55, 0.75)
        y = np.random.uniform(0.3, 0.5)
        circle = Circle((x, y), 0.03, facecolor=COLOR_PURPLE, 
                       edgecolor='white', linewidth=1, alpha=0.7)
        ax.add_patch(circle)
    
    # 中心点标记
    star1 = Circle((0.4, 0.6), 0.04, facecolor=COLOR_BLUE, 
                  edgecolor='white', linewidth=2)
    star2 = Circle((0.65, 0.4), 0.04, facecolor=COLOR_PURPLE, 
                  edgecolor='white', linewidth=2)
    ax.add_patch(star1)
    ax.add_patch(star2)


if __name__ == "__main__":
    print("开始生成PPT专用Logo...\n")
    
    logos = [
        ("Feature Engine", "", icon_feature_engine, "logo_feature_engine.png"),
        ("Embedding", "", icon_embedding, "logo_embedding.png"),
        ("Feature", "Visualization", icon_feature_visualization, "logo_feature_visualization.png"),
    ]
    
    for title, subtitle, icon_func, filename in logos:
        create_logo(title, subtitle, icon_func, filename)
    
    print(f"\n✅ 完成！共生成 {len(logos)} 个Logo")
    print("所有Logo已保存，可直接用于PPT！")
