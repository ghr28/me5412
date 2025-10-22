import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# 定义颜色方案
colors = {
    'data': '#E3F2FD',      # 浅蓝色 - 数据
    'model': '#F3E5F5',     # 浅紫色 - 模型
    'eval': '#E8F5E8',      # 浅绿色 - 评估
    'deploy': '#FFF3E0',    # 浅橙色 - 部署
    'arrow': '#666666'      # 箭头颜色
}

# 标题
ax.text(5, 7.5, '神经康复步态分析AI系统 - 研究方法Pipeline', 
        fontsize=20, fontweight='bold', ha='center')

# 第一层：数据生成与处理
# 1. 数学建模
box1 = FancyBboxPatch((0.2, 6), 1.6, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['data'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box1)
ax.text(1, 6.4, '数学建模\n步态仿真', fontsize=11, ha='center', fontweight='bold')
ax.text(1, 6.1, '4种疾病模式\n400个样本', fontsize=9, ha='center')

# 2. GAN训练
box2 = FancyBboxPatch((2.2, 6), 1.6, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['model'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box2)
ax.text(3, 6.4, '条件GAN\n训练', fontsize=11, ha='center', fontweight='bold')
ax.text(3, 6.1, '生成器+判别器\n5个epochs', fontsize=9, ha='center')

# 3. 数据增强
box3 = FancyBboxPatch((4.2, 6), 1.6, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['data'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box3)
ax.text(5, 6.4, '合成数据\n生成', fontsize=11, ha='center', fontweight='bold')
ax.text(5, 6.1, '400个合成样本\n100%数据增强', fontsize=9, ha='center')

# 4. 数据分割
box4 = FancyBboxPatch((6.2, 6), 1.6, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['data'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box4)
ax.text(7, 6.4, '数据分割', fontsize=11, ha='center', fontweight='bold')
ax.text(7, 6.1, '训练70%\n验证15%\n测试15%', fontsize=9, ha='center')

# 5. 部署应用
box5 = FancyBboxPatch((8.2, 6), 1.6, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['deploy'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box5)
ax.text(9, 6.4, '临床部署', fontsize=11, ha='center', fontweight='bold')
ax.text(9, 6.1, 'Web API\n实时诊断', fontsize=9, ha='center')

# 第二层：模型架构
# TCN模块
box6 = FancyBboxPatch((1, 4.5), 2, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['model'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box6)
ax.text(2, 4.9, 'TCN时序卷积网络', fontsize=11, ha='center', fontweight='bold')
ax.text(2, 4.6, '时序特征提取\n卷积核=3, 通道[64,128]', fontsize=9, ha='center')

# Transformer模块
box7 = FancyBboxPatch((3.5, 4.5), 2, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['model'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box7)
ax.text(4.5, 4.9, 'Transformer编码器', fontsize=11, ha='center', fontweight='bold')
ax.text(4.5, 4.6, '自注意力机制\n8头, 2层', fontsize=9, ha='center')

# 混合分类器
box8 = FancyBboxPatch((6, 4.5), 2, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['model'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box8)
ax.text(7, 4.9, '混合分类器', fontsize=11, ha='center', fontweight='bold')
ax.text(7, 4.6, '特征融合\n4类疾病输出', fontsize=9, ha='center')

# 第三层：评估与结果
# 性能评估
box9 = FancyBboxPatch((1.5, 3), 2.5, 0.8, 
                      boxstyle="round,pad=0.1", 
                      facecolor=colors['eval'], 
                      edgecolor='black', linewidth=1.5)
ax.add_patch(box9)
ax.text(2.75, 3.4, '性能评估', fontsize=11, ha='center', fontweight='bold')
ax.text(2.75, 3.1, '准确率75.8% | 精确率81.5%\nF1分数75.0%', fontsize=9, ha='center')

# 混淆矩阵
box10 = FancyBboxPatch((4.5, 3), 2.5, 0.8, 
                       boxstyle="round,pad=0.1", 
                       facecolor=colors['eval'], 
                       edgecolor='black', linewidth=1.5)
ax.add_patch(box10)
ax.text(5.75, 3.4, '分类别分析', fontsize=11, ha='center', fontweight='bold')
ax.text(5.75, 3.1, '脑瘫97.1% | 帕金森92.6%\n健康62.1% | 脑卒中48.3%', fontsize=9, ha='center')

# 数据流向箭头 - 第一层
for i in range(4):
    arrow = ConnectionPatch((1.8 + i*2, 6.4), (2.2 + i*2, 6.4), 
                           "data", "data", 
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc=colors['arrow'], 
                           color=colors['arrow'], linewidth=2)
    ax.add_patch(arrow)

# 垂直箭头：数据到模型
arrow_v1 = ConnectionPatch((3, 6), (2, 5.3), 
                          "data", "data", 
                          arrowstyle="->", shrinkA=5, shrinkB=5, 
                          mutation_scale=20, fc=colors['arrow'], 
                          color=colors['arrow'], linewidth=2)
ax.add_patch(arrow_v1)

# 模型内部连接箭头
arrow_m1 = ConnectionPatch((3, 4.9), (3.5, 4.9), 
                          "data", "data", 
                          arrowstyle="->", shrinkA=5, shrinkB=5, 
                          mutation_scale=20, fc=colors['arrow'], 
                          color=colors['arrow'], linewidth=2)
ax.add_patch(arrow_m1)

arrow_m2 = ConnectionPatch((5.5, 4.9), (6, 4.9), 
                          "data", "data", 
                          arrowstyle="->", shrinkA=5, shrinkB=5, 
                          mutation_scale=20, fc=colors['arrow'], 
                          color=colors['arrow'], linewidth=2)
ax.add_patch(arrow_m2)

# 模型到评估箭头
arrow_v2 = ConnectionPatch((5, 4.5), (4, 3.8), 
                          "data", "data", 
                          arrowstyle="->", shrinkA=5, shrinkB=5, 
                          mutation_scale=20, fc=colors['arrow'], 
                          color=colors['arrow'], linewidth=2)
ax.add_patch(arrow_v2)

# 评估结果连接
arrow_e1 = ConnectionPatch((4, 3.4), (4.5, 3.4), 
                          "data", "data", 
                          arrowstyle="->", shrinkA=5, shrinkB=5, 
                          mutation_scale=20, fc=colors['arrow'], 
                          color=colors['arrow'], linewidth=2)
ax.add_patch(arrow_e1)

# 添加技术指标文本框
tech_box = FancyBboxPatch((0.5, 1.5), 4, 1, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#F5F5F5', 
                          edgecolor='black', linewidth=1)
ax.add_patch(tech_box)
ax.text(2.5, 2.2, '核心技术指标', fontsize=12, ha='center', fontweight='bold')
ax.text(2.5, 1.9, '• 数据规模: 400→800个样本 (100%增强)', fontsize=10, ha='center')
ax.text(2.5, 1.7, '• 模型架构: TCN + Transformer混合设计', fontsize=10, ha='center')
ax.text(2.5, 1.5, '• 训练效率: 28.9秒端到端pipeline', fontsize=10, ha='center')

# 添加应用价值文本框
app_box = FancyBboxPatch((5.5, 1.5), 4, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor='#F0F8FF', 
                         edgecolor='black', linewidth=1)
ax.add_patch(app_box)
ax.text(7.5, 2.2, '临床应用价值', fontsize=12, ha='center', fontweight='bold')
ax.text(7.5, 1.9, '• 诊断效率: 数分钟 vs 传统数小时', fontsize=10, ha='center')
ax.text(7.5, 1.7, '• 成本降低: $50-100 vs 传统$500-1000', fontsize=10, ha='center')
ax.text(7.5, 1.5, '• 客观评估: 消除主观判断偏差', fontsize=10, ha='center')

# 添加图例
legend_elements = [
    mpatches.Patch(color=colors['data'], label='数据处理'),
    mpatches.Patch(color=colors['model'], label='模型训练'),
    mpatches.Patch(color=colors['eval'], label='性能评估'),
    mpatches.Patch(color=colors['deploy'], label='部署应用')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig('/home/ghr/me5412/docs/research_methodology_pipeline.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 研究方法Pipeline流程图已生成: docs/research_methodology_pipeline.png")
