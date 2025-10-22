import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建时间序列
t = np.linspace(0, 1, 100)

# 定义四种疾病的步态函数
def healthy_gait(t):
    """健康步态：规律对称的正弦波模式"""
    left_leg_x = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(100)
    left_leg_y = np.sin(2 * np.pi * t + np.pi) + 0.1 * np.random.randn(100)
    left_leg_z = 0.5 * np.sin(4 * np.pi * t) + 0.05 * np.random.randn(100)
    right_leg_x = np.sin(2 * np.pi * t + np.pi) + 0.1 * np.random.randn(100)
    right_leg_y = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(100)
    right_leg_z = 0.5 * np.sin(4 * np.pi * t + np.pi) + 0.05 * np.random.randn(100)
    return left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z

def parkinsons_gait(t):
    """帕金森步态：幅度衰减，不规律，震颤"""
    tremor = 0.3 * np.sin(20 * np.pi * t)  # 高频震颤
    left_leg_x = 0.7 * np.sin(2 * np.pi * t) + tremor + 0.2 * np.random.randn(100)
    left_leg_y = 0.7 * np.sin(2 * np.pi * t + np.pi) + tremor + 0.2 * np.random.randn(100)
    left_leg_z = 0.3 * np.sin(4 * np.pi * t) + 0.1 * tremor + 0.1 * np.random.randn(100)
    right_leg_x = 0.7 * np.sin(2 * np.pi * t + np.pi) + tremor + 0.2 * np.random.randn(100)
    right_leg_y = 0.7 * np.sin(2 * np.pi * t) + tremor + 0.2 * np.random.randn(100)
    right_leg_z = 0.3 * np.sin(4 * np.pi * t + np.pi) + 0.1 * tremor + 0.1 * np.random.randn(100)
    return left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z

def cerebral_palsy_gait(t):
    """脑瘫步态：痉挛性，不对称，高频抖动"""
    spasticity = 0.3 * np.sin(6 * np.pi * t)  # 痉挛成分
    left_leg_x = np.sin(2 * np.pi * t) + spasticity + 0.2 * np.random.randn(100)
    left_leg_y = 0.6 * np.sin(2 * np.pi * t + np.pi) + 0.2 * np.random.randn(100)  # 不对称
    left_leg_z = 0.4 * np.sin(4 * np.pi * t) + 0.2 * np.sin(8 * np.pi * t) + 0.1 * np.random.randn(100)
    right_leg_x = 0.8 * np.sin(2 * np.pi * t + np.pi) + 0.5 * spasticity + 0.2 * np.random.randn(100)
    right_leg_y = 0.4 * np.sin(2 * np.pi * t) + 0.2 * np.random.randn(100)  # 严重不对称
    right_leg_z = 0.3 * np.sin(4 * np.pi * t + np.pi) + 0.1 * np.random.randn(100)
    return left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z

def stroke_gait(t):
    """脑卒中步态：一侧偏瘫，明显不对称"""
    # 健侧（左侧）相对正常
    left_leg_x = 0.8 * np.sin(2 * np.pi * t) + 0.15 * np.random.randn(100)
    left_leg_y = 0.8 * np.sin(2 * np.pi * t + np.pi) + 0.15 * np.random.randn(100)
    left_leg_z = 0.4 * np.sin(4 * np.pi * t) + 0.1 * np.random.randn(100)
    
    # 患侧（右侧）明显受损
    right_leg_x = 0.4 * np.sin(2 * np.pi * t + np.pi) + 0.3 * np.random.randn(100)  # 幅度大幅衰减
    right_leg_y = 0.3 * np.sin(2 * np.pi * t) + 0.3 * np.random.randn(100)  # 更严重衰减
    right_leg_z = 0.2 * np.sin(4 * np.pi * t + np.pi) + 0.2 * np.random.randn(100)  # 垂直运动受限
    return left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z

# 设置随机种子以保证可重现性
np.random.seed(42)

# 生成四种疾病的数据
healthy_data = healthy_gait(t)
parkinsons_data = parkinsons_gait(t)
cerebral_palsy_data = cerebral_palsy_gait(t)
stroke_data = stroke_gait(t)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Four Disease Gait Pattern Functions\n(Healthy, Parkinson\'s, Cerebral Palsy, Stroke)', 
             fontsize=16, fontweight='bold')

diseases = [
    ('Healthy Control', healthy_data, 'green'),
    ('Parkinson\'s Disease', parkinsons_data, 'orange'),
    ('Cerebral Palsy', cerebral_palsy_data, 'red'),
    ('Stroke', stroke_data, 'purple')
]

# 绘制每种疾病的步态模式
for idx, (disease_name, data, color) in enumerate(diseases):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    left_leg_x, left_leg_y, left_leg_z, right_leg_x, right_leg_y, right_leg_z = data
    
    # 绘制左腿和右腿的xyz三轴加速度曲线
    ax.plot(t, left_leg_x, label='Left Leg X', color=color, alpha=0.8, linewidth=2)
    ax.plot(t, left_leg_y, label='Left Leg Y', color=color, alpha=0.6, linewidth=2, linestyle='--')
    ax.plot(t, left_leg_z, label='Left Leg Z', color=color, alpha=0.4, linewidth=2, linestyle=':')
    ax.plot(t, right_leg_x, label='Right Leg X', color='blue', alpha=0.8, linewidth=2)
    ax.plot(t, right_leg_y, label='Right Leg Y', color='blue', alpha=0.6, linewidth=2, linestyle='--')
    ax.plot(t, right_leg_z, label='Right Leg Z', color='blue', alpha=0.4, linewidth=2, linestyle=':')
    
    ax.set_title(f'{disease_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Gait Cycle)', fontsize=12)
    ax.set_ylabel('Acceleration (normalized)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(-3, 3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('/home/ghr/me5412/docs/four_disease_gait_functions.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 创建第二个图：显示数学函数表达式
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
ax2.axis('off')

# 添加标题
ax2.text(0.5, 0.95, 'Mathematical Models for Four Disease Gait Patterns', 
         fontsize=18, fontweight='bold', ha='center', transform=ax2.transAxes)

# 定义函数表达式
function_expressions = [
    {
        'name': '1. Healthy Control Gait Function',
        'color': 'green',
        'equations': [
            'Left Leg X: x₁(t) = sin(2πt) + ε₁(t)    [Regular, symmetric]',
            'Left Leg Y: y₁(t) = sin(2πt + π) + ε₁(t)  [Phase shifted]',
            'Left Leg Z: z₁(t) = 0.5·sin(4πt) + ε₁(t)  [Vertical movement]',
            'Right Leg X: x₂(t) = sin(2πt + π) + ε₂(t)  [Opposite phase]',
            'Right Leg Y: y₂(t) = sin(2πt) + ε₂(t)    [Symmetric pattern]',
            'Right Leg Z: z₂(t) = 0.5·sin(4πt + π) + ε₂(t)  [Vertical sync]',
            'Noise: ε(t) ~ N(0, 0.1²)  [Low noise level]'
        ]
    },
    {
        'name': '2. Parkinson\'s Disease Gait Function',
        'color': 'orange',
        'equations': [
            'Left Leg X: x₁(t) = 0.7·sin(2πt) + 0.3·sin(20πt) + ε₁(t)',
            'Left Leg Y: y₁(t) = 0.7·sin(2πt + π) + 0.3·sin(20πt) + ε₁(t)',
            'Left Leg Z: z₁(t) = 0.3·sin(4πt) + 0.1·sin(20πt) + ε₁(t)',
            'Right Leg X: x₂(t) = 0.7·sin(2πt + π) + 0.3·sin(20πt) + ε₂(t)',
            'Right Leg Y: y₂(t) = 0.7·sin(2πt) + 0.3·sin(20πt) + ε₂(t)',
            'Right Leg Z: z₂(t) = 0.3·sin(4πt + π) + 0.1·sin(20πt) + ε₂(t)',
            'Features: Amplitude reduction (0.7×), tremor (20πt), high noise'
        ]
    },
    {
        'name': '3. Cerebral Palsy Gait Function',
        'color': 'red',
        'equations': [
            'Left Leg X: x₁(t) = sin(2πt) + 0.3·sin(6πt) + ε₁(t)',
            'Left Leg Y: y₁(t) = 0.6·sin(2πt + π) + ε₁(t)  [Asymmetric]',
            'Left Leg Z: z₁(t) = 0.4·sin(4πt) + 0.2·sin(8πt) + ε₁(t)',
            'Right Leg X: x₂(t) = 0.8·sin(2πt + π) + 0.5·sin(6πt) + ε₂(t)',
            'Right Leg Y: y₂(t) = 0.4·sin(2πt) + ε₂(t)  [Severe asymmetry]',
            'Right Leg Z: z₂(t) = 0.3·sin(4πt + π) + ε₂(t)',
            'Features: Spasticity component (6πt), bilateral asymmetry'
        ]
    },
    {
        'name': '4. Stroke Gait Function (Hemiplegia)',
        'color': 'purple',
        'equations': [
            'Unaffected Side (Left):',
            '  x₁(t) = 0.8·sin(2πt) + ε₁(t)  [Relatively normal]',
            '  y₁(t) = 0.8·sin(2πt + π) + ε₁(t)',
            '  z₁(t) = 0.4·sin(4πt) + ε₁(t)',
            'Affected Side (Right):',
            '  x₂(t) = 0.4·sin(2πt + π) + ε₂(t)  [Severe impairment]',
            '  y₂(t) = 0.3·sin(2πt) + ε₂(t)  [Marked weakness]',
            '  z₂(t) = 0.2·sin(4πt + π) + ε₂(t)  [Limited vertical]',
            'Features: Unilateral weakness, compensatory patterns'
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
        ax2.text(0.1, y_pos, eq, fontsize=11, fontfamily='monospace',
                 transform=ax2.transAxes)
    y_pos -= 0.03

# 添加说明
ax2.text(0.05, 0.15, 'Mathematical Notation:', fontsize=12, fontweight='bold',
         transform=ax2.transAxes)
ax2.text(0.1, 0.12, 't: normalized time in gait cycle [0, 1]', fontsize=11,
         transform=ax2.transAxes)
ax2.text(0.1, 0.09, 'ε(t): Gaussian noise representing natural variability', fontsize=11,
         transform=ax2.transAxes)
ax2.text(0.1, 0.06, 'π: pi constant (3.14159...)', fontsize=11,
         transform=ax2.transAxes)
ax2.text(0.1, 0.03, 'sin(): sine function for periodic gait patterns', fontsize=11,
         transform=ax2.transAxes)

# 保存第二个图
plt.savefig('/home/ghr/me5412/docs/disease_gait_mathematical_functions.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ 四种疾病步态函数图片已生成:")
print("1. four_disease_gait_functions.png - 时域信号对比图")
print("2. disease_gait_mathematical_functions.png - 数学函数表达式")
