import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建时间序列
t = np.linspace(0, 2, 200)  # 2个步态周期

def healthy_gait_normalized(t):
    """健康步态：标准化相对运动幅度"""
    # 左腿运动模式
    left_leg_x = np.sin(2 * np.pi * t) + 0.05 * np.random.randn(len(t))           # 前后运动
    left_leg_y = np.sin(2 * np.pi * t + np.pi) + 0.05 * np.random.randn(len(t))   # 左右运动
    left_leg_z = 0.5 * np.sin(4 * np.pi * t) + 0.03 * np.random.randn(len(t))     # 垂直运动
    
    # 右腿运动模式 (与左腿相位相反)
    right_leg_x = np.sin(2 * np.pi * t + np.pi) + 0.05 * np.random.randn(len(t))  # 前后运动
    right_leg_y = np.sin(2 * np.pi * t) + 0.05 * np.random.randn(len(t))          # 左右运动
    right_leg_z = 0.5 * np.sin(4 * np.pi * t + np.pi) + 0.03 * np.random.randn(len(t))  # 垂直运动
    
    return left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z

def parkinsons_gait_normalized(t):
    """帕金森病步态：幅度减小 + 震颤"""
    # 震颤成分 (高频小幅度)
    tremor = 0.15 * np.sin(20 * np.pi * t)
    
    # 左腿运动模式 (幅度减小70%)
    left_leg_x = 0.7 * np.sin(2 * np.pi * t) + tremor + 0.1 * np.random.randn(len(t))
    left_leg_y = 0.7 * np.sin(2 * np.pi * t + np.pi) + tremor + 0.1 * np.random.randn(len(t))
    left_leg_z = 0.35 * np.sin(4 * np.pi * t) + 0.5 * tremor + 0.08 * np.random.randn(len(t))
    
    # 右腿运动模式 (同样幅度减小)
    right_leg_x = 0.7 * np.sin(2 * np.pi * t + np.pi) + tremor + 0.1 * np.random.randn(len(t))
    right_leg_y = 0.7 * np.sin(2 * np.pi * t) + tremor + 0.1 * np.random.randn(len(t))
    right_leg_z = 0.35 * np.sin(4 * np.pi * t + np.pi) + 0.5 * tremor + 0.08 * np.random.randn(len(t))
    
    return left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z

def cerebral_palsy_gait_normalized(t):
    """脑瘫步态：痉挛性 + 不对称性"""
    # 痉挛成分 (中频高幅度)
    spasticity = 0.2 * np.sin(6 * np.pi * t)
    
    # 左腿运动模式 (痉挛影响)
    left_leg_x = np.sin(2 * np.pi * t) + spasticity + 0.15 * np.random.randn(len(t))
    left_leg_y = 0.6 * np.sin(2 * np.pi * t + np.pi) + 0.1 * np.random.randn(len(t))  # 不对称
    left_leg_z = 0.4 * np.sin(4 * np.pi * t) + 0.15 * np.sin(8 * np.pi * t) + 0.1 * np.random.randn(len(t))
    
    # 右腿运动模式 (更严重的不对称)
    right_leg_x = 0.8 * np.sin(2 * np.pi * t + np.pi) + 0.7 * spasticity + 0.15 * np.random.randn(len(t))
    right_leg_y = 0.4 * np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))  # 严重不对称
    right_leg_z = 0.3 * np.sin(4 * np.pi * t + np.pi) + 0.1 * np.random.randn(len(t))
    
    return left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z

def stroke_gait_normalized(t):
    """脑卒中步态：偏瘫 + 明显不对称"""
    # 健侧 (左侧) - 相对正常但有补偿
    left_leg_x = 0.8 * np.sin(2 * np.pi * t) + 0.08 * np.random.randn(len(t))
    left_leg_y = 0.8 * np.sin(2 * np.pi * t + np.pi) + 0.08 * np.random.randn(len(t))
    left_leg_z = 0.4 * np.sin(4 * np.pi * t) + 0.06 * np.random.randn(len(t))
    
    # 患侧 (右侧) - 明显受损
    right_leg_x = 0.3 * np.sin(2 * np.pi * t + np.pi) + 0.2 * np.random.randn(len(t))  # 严重减弱
    right_leg_y = 0.2 * np.sin(2 * np.pi * t) + 0.25 * np.random.randn(len(t))         # 控制困难
    right_leg_z = 0.15 * np.sin(4 * np.pi * t + np.pi) + 0.15 * np.random.randn(len(t))  # 抬腿困难
    
    return left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z

# 设置随机种子
np.random.seed(42)

# 生成四种疾病的标准化数据
healthy_data = healthy_gait_normalized(t)
parkinsons_data = parkinsons_gait_normalized(t)
cerebral_palsy_data = cerebral_palsy_gait_normalized(t)
stroke_data = stroke_gait_normalized(t)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Normalized Gait Movement Patterns\n(Relative Motion Amplitude for Four Disease Types)', 
             fontsize=16, fontweight='bold')

diseases = [
    ('Healthy Control', healthy_data, 'green'),
    ('Parkinson Disease', parkinsons_data, 'orange'),
    ('Cerebral Palsy', cerebral_palsy_data, 'red'),
    ('Stroke (Hemiplegia)', stroke_data, 'purple')
]

# 绘制每种疾病的步态模式
for idx, (disease_name, data, color) in enumerate(diseases):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z = data
    
    # 绘制左腿和右腿的xyz三轴相对运动幅度
    ax.plot(t, left_leg_x, label='Left Leg - Anterior/Posterior', 
            color=color, alpha=0.9, linewidth=2.5)
    ax.plot(t, left_leg_y, label='Left Leg - Medial/Lateral', 
            color=color, alpha=0.7, linewidth=2, linestyle='--')
    ax.plot(t, left_leg_z, label='Left Leg - Vertical', 
            color=color, alpha=0.5, linewidth=1.5, linestyle=':')
    
    ax.plot(t, right_leg_x, label='Right Leg - Anterior/Posterior', 
            color='blue', alpha=0.9, linewidth=2.5)
    ax.plot(t, right_leg_y, label='Right Leg - Medial/Lateral', 
            color='blue', alpha=0.7, linewidth=2, linestyle='--')
    ax.plot(t, right_leg_z, label='Right Leg - Vertical', 
            color='blue', alpha=0.5, linewidth=1.5, linestyle=':')
    
    # 添加零线参考
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    ax.set_title(f'{disease_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Gait Cycles)', fontsize=12)
    ax.set_ylabel('Normalized Movement Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(-1.5, 1.5)
    
    # 添加疾病特征说明
    if disease_name == 'Healthy Control':
        ax.text(0.05, 1.2, 'Features:\n• Regular patterns\n• Symmetric L/R\n• Amplitude = 1.0', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    elif disease_name == 'Parkinson Disease':
        ax.text(0.05, 1.2, 'Features:\n• Reduced amplitude (0.7×)\n• High-freq tremor\n• Bradykinesia', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    elif disease_name == 'Cerebral Palsy':
        ax.text(0.05, 1.2, 'Features:\n• Spastic components\n• L/R asymmetry\n• Irregular patterns', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    else:  # Stroke
        ax.text(0.05, 1.2, 'Features:\n• Severe L/R asymmetry\n• Affected side weak\n• Compensatory gait', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

plt.tight_layout()
plt.savefig('/home/ghr/me5412/docs/normalized_gait_movement_patterns.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated: normalized_gait_movement_patterns.png")

# 创建第二个图：数学函数表达式 (更新为标准化版本)
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
ax2.axis('off')

# 添加标题
ax2.text(0.5, 0.95, 'Mathematical Models: Normalized Gait Movement Patterns', 
         fontsize=18, fontweight='bold', ha='center', transform=ax2.transAxes)

# 定义更新后的函数表达式
function_expressions = [
    {
        'name': '1. Healthy Control Gait Pattern (Normalized)',
        'color': 'green',
        'equations': [
            'Left Leg X:  x₁(t) = sin(2πt) + ε₁(t)           [Regular A-P motion]',
            'Left Leg Y:  y₁(t) = sin(2πt + π) + ε₁(t)       [Regular M-L motion]',
            'Left Leg Z:  z₁(t) = 0.5·sin(4πt) + ε₁(t)       [Vertical motion]',
            'Right Leg X: x₂(t) = sin(2πt + π) + ε₂(t)       [Opposite phase]',
            'Right Leg Y: y₂(t) = sin(2πt) + ε₂(t)           [Symmetric pattern]',
            'Right Leg Z: z₂(t) = 0.5·sin(4πt + π) + ε₂(t)   [Vertical sync]',
            'Noise: ε(t) ~ N(0, 0.05²)  [Low variability]'
        ]
    },
    {
        'name': '2. Parkinson Disease Gait Pattern (Normalized)',
        'color': 'orange',
        'equations': [
            'Left Leg X:  x₁(t) = 0.7·sin(2πt) + 0.15·sin(20πt) + ε₁(t)',
            'Left Leg Y:  y₁(t) = 0.7·sin(2πt + π) + 0.15·sin(20πt) + ε₁(t)',
            'Left Leg Z:  z₁(t) = 0.35·sin(4πt) + 0.075·sin(20πt) + ε₁(t)',
            'Right Leg X: x₂(t) = 0.7·sin(2πt + π) + 0.15·sin(20πt) + ε₂(t)',
            'Right Leg Y: y₂(t) = 0.7·sin(2πt) + 0.15·sin(20πt) + ε₂(t)',
            'Right Leg Z: z₂(t) = 0.35·sin(4πt + π) + 0.075·sin(20πt) + ε₂(t)',
            'Features: Amplitude reduction (0.7×), tremor (20πt), bradykinesia'
        ]
    },
    {
        'name': '3. Cerebral Palsy Gait Pattern (Normalized)',
        'color': 'red',
        'equations': [
            'Left Leg X:  x₁(t) = sin(2πt) + 0.2·sin(6πt) + ε₁(t)',
            'Left Leg Y:  y₁(t) = 0.6·sin(2πt + π) + ε₁(t)    [Asymmetric]',
            'Left Leg Z:  z₁(t) = 0.4·sin(4πt) + 0.15·sin(8πt) + ε₁(t)',
            'Right Leg X: x₂(t) = 0.8·sin(2πt + π) + 0.14·sin(6πt) + ε₂(t)',
            'Right Leg Y: y₂(t) = 0.4·sin(2πt) + ε₂(t)        [Severe asymmetry]',
            'Right Leg Z: z₂(t) = 0.3·sin(4πt + π) + ε₂(t)',
            'Features: Spasticity (6πt), bilateral asymmetry, incoordination'
        ]
    },
    {
        'name': '4. Stroke Gait Pattern (Hemiplegia, Normalized)',
        'color': 'purple',
        'equations': [
            'Unaffected Side (Left):',
            '  x₁(t) = 0.8·sin(2πt) + ε₁(t)        [Compensatory]',
            '  y₁(t) = 0.8·sin(2πt + π) + ε₁(t)    [Relatively normal]',
            '  z₁(t) = 0.4·sin(4πt) + ε₁(t)',
            'Affected Side (Right):',
            '  x₂(t) = 0.3·sin(2πt + π) + ε₂(t)    [Severe weakness]',
            '  y₂(t) = 0.2·sin(2πt) + ε₂(t)        [Poor control]',
            '  z₂(t) = 0.15·sin(4πt + π) + ε₂(t)   [Limited lifting]',
            'Features: Unilateral paresis, marked L/R asymmetry'
        ]
    }
]

# 绘制函数表达式
y_pos = 0.85
for func_info in function_expressions:
    # 疾病名称
    ax2.text(0.05, y_pos, func_info['name'], fontsize=14, fontweight='bold', 
             color=func_info['color'], transform=ax2.transAxes)
    y_pos -= 0.03
    
    # 方程式
    for eq in func_info['equations']:
        y_pos -= 0.025
        ax2.text(0.1, y_pos, eq, fontsize=10, fontfamily='monospace',
                 transform=ax2.transAxes)
    y_pos -= 0.03

# 添加说明
ax2.text(0.05, 0.15, 'Mathematical Notation:', fontsize=12, fontweight='bold',
         transform=ax2.transAxes)
ax2.text(0.1, 0.12, 't: normalized time in gait cycle [0, 1]', fontsize=11,
         transform=ax2.transAxes)
ax2.text(0.1, 0.09, 'ε(t): Gaussian noise (natural variability)', fontsize=11,
         transform=ax2.transAxes)
ax2.text(0.1, 0.06, 'A-P: Anterior-Posterior, M-L: Medial-Lateral', fontsize=11,
         transform=ax2.transAxes)
ax2.text(0.1, 0.03, 'All values normalized to [-1, 1] range', fontsize=11,
         transform=ax2.transAxes, fontweight='bold', color='red')

plt.savefig('/home/ghr/me5412/docs/normalized_gait_mathematical_functions.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated: normalized_gait_mathematical_functions.png")

# 创建对比图：原始混合版 vs 统一标准化版
fig3, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(16, 10))

# 上图：原始混合版本
t_demo = np.linspace(0, 1, 100)

# 原始混合数据 (前5维标准化 + 第6维真实加速度)
original_mixed = np.column_stack([
    np.sin(2 * np.pi * t_demo),                           # 标准化 [-1,1]
    np.sin(2 * np.pi * t_demo + np.pi),                   # 标准化 [-1,1]
    0.5 * np.sin(4 * np.pi * t_demo),                     # 标准化 [-0.5,0.5]
    np.sin(2 * np.pi * t_demo + np.pi),                   # 标准化 [-1,1]
    0.5 * np.sin(4 * np.pi * t_demo + np.pi),             # 标准化 [-0.5,0.5]
    9.8 + 0.2 * np.sin(2 * np.pi * t_demo)                # 真实加速度 [9.6,10.0]
])

ax_top.plot(t_demo, original_mixed[:, 0], 'b-', linewidth=2, label='Dimension 1-5 (Normalized)')
ax_top.plot(t_demo, original_mixed[:, 1], 'b--', linewidth=2, alpha=0.7)
ax_top.plot(t_demo, original_mixed[:, 2], 'b:', linewidth=2, alpha=0.7)
ax_top.plot(t_demo, original_mixed[:, 3], 'g-', linewidth=2, alpha=0.7)
ax_top.plot(t_demo, original_mixed[:, 4], 'g--', linewidth=2, alpha=0.7)

# 右侧Y轴显示第6维
ax_top_right = ax_top.twinx()
ax_top_right.plot(t_demo, original_mixed[:, 5], 'r-', linewidth=3, label='Dimension 6 (Real Acceleration)')
ax_top_right.set_ylabel('Acceleration (m/s²)', fontsize=12, color='red')
ax_top_right.tick_params(axis='y', labelcolor='red')

ax_top.set_title('Original Model: Mixed Physical Quantities\n(Dimensions 1-5: Normalized [-1,1] + Dimension 6: Real Acceleration [9.6-10.0 m/s²])', 
                fontsize=14, fontweight='bold')
ax_top.set_ylabel('Normalized Amplitude', fontsize=12)
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc='upper left')
ax_top_right.legend(loc='upper right')
ax_top.text(0.02, 0.8, 'PROBLEM:\nMixed units make\ncomparison difficult', 
           transform=ax_top.transAxes, fontsize=11,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 下图：统一标准化版本
healthy_norm = healthy_gait_normalized(t_demo)
unified_data = np.column_stack(healthy_norm)

ax_bottom.plot(t_demo, unified_data[:, 0], 'b-', linewidth=2, label='Left Leg A-P')
ax_bottom.plot(t_demo, unified_data[:, 1], 'b--', linewidth=2, label='Left Leg M-L')
ax_bottom.plot(t_demo, unified_data[:, 2], 'b:', linewidth=2, label='Left Leg Vertical')
ax_bottom.plot(t_demo, unified_data[:, 3], 'g-', linewidth=2, label='Right Leg A-P')
ax_bottom.plot(t_demo, unified_data[:, 4], 'g--', linewidth=2, label='Right Leg M-L')
ax_bottom.plot(t_demo, unified_data[:, 5], 'g:', linewidth=2, label='Right Leg Vertical')

ax_bottom.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax_bottom.set_title('Improved Model: Unified Normalized Movement Patterns\n(All Dimensions: Relative Motion Amplitude [-1,1])', 
                   fontsize=14, fontweight='bold')
ax_bottom.set_xlabel('Time (Gait Cycle)', fontsize=12)
ax_bottom.set_ylabel('Normalized Movement Amplitude', fontsize=12)
ax_bottom.grid(True, alpha=0.3)
ax_bottom.legend(fontsize=10)
ax_bottom.set_ylim(-1.2, 1.2)
ax_bottom.text(0.02, 0.8, 'SOLUTION:\nUnified scale enables\ndirect comparison', 
              transform=ax_bottom.transAxes, fontsize=11,
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.suptitle('Model Comparison: Mixed vs Unified Physical Quantities', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ghr/me5412/docs/model_comparison_mixed_vs_unified.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated: model_comparison_mixed_vs_unified.png")

print("\n✅ Successfully generated normalized gait model charts:")
print("1. Four disease types with unified normalized patterns")
print("2. Updated mathematical function expressions")
print("3. Comparison between mixed and unified models")
print("\n💡 Improvements:")
print("- Unified physical quantities: All relative motion amplitude [-1,1]")
print("- Easy comparison: Same scale for direct comparison")
print("- Clearer disease features: Identified by amplitude and pattern differences")
print("- Model consistency: Eliminated conceptual confusion of mixed quantities")
