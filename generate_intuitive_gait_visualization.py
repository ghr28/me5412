import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建时间序列
t = np.linspace(0, 2, 200)  # 2个步态周期，更清晰

# 定义更直观的步态参数函数
def healthy_gait_intuitive(t):
    """健康步态 - 使用关节角度和步态参数"""
    knee_angle = 60 + 30 * np.sin(2 * np.pi * t)  # 膝关节角度 60°±30°
    step_length = 70 + 5 * np.sin(2 * np.pi * t)  # 步长 70±5cm
    gait_speed = 120 + 10 * np.sin(2 * np.pi * t)  # 步频 120±10步/分钟
    return knee_angle, step_length, gait_speed

def parkinsons_gait_intuitive(t):
    """帕金森病步态 - 小步态、僵硬"""
    n_points = len(t)
    knee_angle = 60 + 15 * np.sin(2 * np.pi * t) + 3 * np.random.randn(n_points)
    step_length = 45 + 8 * np.sin(2 * np.pi * t) + 2 * np.random.randn(n_points)
    gait_speed = 80 + 15 * np.sin(2 * np.pi * t) + 5 * np.random.randn(n_points)
    return knee_angle, step_length, gait_speed

def cerebral_palsy_gait_intuitive(t):
    """脑瘫步态 - 痉挛性、不协调"""
    knee_angle = 60 + 40 * np.sin(2 * np.pi * t) + 15 * np.sin(6 * np.pi * t)  # 痉挛成分
    step_length = 55 + 20 * np.sin(2 * np.pi * t) + 10 * np.sin(4 * np.pi * t)  # 不规律
    gait_speed = 90 + 25 * np.sin(2 * np.pi * t) + 15 * np.sin(6 * np.pi * t)  # 不协调
    return knee_angle, step_length, gait_speed

def stroke_gait_intuitive(t):
    """脑卒中步态 - 偏瘫，不对称"""
    # 健侧（相对正常）
    knee_angle_healthy = 60 + 25 * np.sin(2 * np.pi * t)
    step_length_healthy = 65 + 5 * np.sin(2 * np.pi * t)
    
    # 患侧（明显受损）
    knee_angle_affected = 50 + 10 * np.sin(2 * np.pi * t)  # 活动度受限
    step_length_affected = 35 + 8 * np.sin(2 * np.pi * t)  # 明显短步
    
    # 平均步频
    gait_speed = 70 + 10 * np.sin(np.pi * t)  # 整体变慢
    
    return knee_angle_healthy, knee_angle_affected, step_length_healthy, step_length_affected, gait_speed

# 设置随机种子
np.random.seed(42)

# 创建图形 - 三行对比
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Intuitive Gait Analysis: Joint Angles & Gait Parameters\\n(More Understandable for Clinical Application)', 
             fontsize=16, fontweight='bold')

diseases = ['Healthy Control', "Parkinson's Disease", 'Cerebral Palsy', 'Stroke (Hemiplegia)']
colors = ['green', 'orange', 'red', 'purple']

# 第一行：膝关节角度
for i, (disease, color) in enumerate(zip(diseases, colors)):
    ax = axes[0, i]
    
    if disease == 'Healthy Control':
        knee_angle, _, _ = healthy_gait_intuitive(t)
        ax.plot(t, knee_angle, color=color, linewidth=2.5, label='Knee Angle')
        ax.text(0.1, 85, 'Normal Range:\\n60° ± 30°', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
    elif disease == "Parkinson's Disease":
        knee_angle, _, _ = parkinsons_gait_intuitive(t)
        ax.plot(t, knee_angle, color=color, linewidth=2.5, label='Knee Angle')
        ax.text(0.1, 85, 'Reduced Range:\\n60° ± 15°\\n+ Tremor', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        
    elif disease == 'Cerebral Palsy':
        knee_angle, _, _ = cerebral_palsy_gait_intuitive(t)
        ax.plot(t, knee_angle, color=color, linewidth=2.5, label='Knee Angle')
        ax.text(0.1, 95, 'Spastic Pattern:\\n60° ± 40°\\n+ High Freq', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
    else:  # Stroke
        knee_healthy, knee_affected, _, _, _ = stroke_gait_intuitive(t)
        ax.plot(t, knee_healthy, color='blue', linewidth=2.5, label='Healthy Side', alpha=0.8)
        ax.plot(t, knee_affected, color=color, linewidth=2.5, label='Affected Side', alpha=0.8)
        ax.text(0.1, 75, 'Asymmetric:\\nHealthy: 60°±25°\\nAffected: 50°±10°', 
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    ax.set_title(f'{disease}\\nKnee Joint Angle', fontsize=12, fontweight='bold')
    ax.set_ylabel('Angle (degrees)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(10, 100)

# 第二行：步长
for i, (disease, color) in enumerate(zip(diseases, colors)):
    ax = axes[1, i]
    
    if disease == 'Healthy Control':
        _, step_length, _ = healthy_gait_intuitive(t)
        ax.plot(t, step_length, color=color, linewidth=2.5, label='Step Length')
        ax.text(0.1, 77, 'Normal:\\n70 ± 5 cm', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
    elif disease == "Parkinson's Disease":
        _, step_length, _ = parkinsons_gait_intuitive(t)
        ax.plot(t, step_length, color=color, linewidth=2.5, label='Step Length')
        ax.text(0.1, 52, 'Short Steps:\\n45 ± 8 cm\\n(Bradykinesia)', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        
    elif disease == 'Cerebral Palsy':
        _, step_length, _ = cerebral_palsy_gait_intuitive(t)
        ax.plot(t, step_length, color=color, linewidth=2.5, label='Step Length')
        ax.text(0.1, 85, 'Irregular:\\n55 ± 20 cm\\n(Spasticity)', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
    else:  # Stroke
        _, _, step_healthy, step_affected, _ = stroke_gait_intuitive(t)
        ax.plot(t, step_healthy, color='blue', linewidth=2.5, label='Healthy Side', alpha=0.8)
        ax.plot(t, step_affected, color=color, linewidth=2.5, label='Affected Side', alpha=0.8)
        ax.text(0.1, 70, 'Asymmetric:\\nHealthy: 65±5cm\\nAffected: 35±8cm', 
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    ax.set_title(f'Step Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Length (cm)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(20, 90)

# 第三行：步频/行走速度
for i, (disease, color) in enumerate(zip(diseases, colors)):
    ax = axes[2, i]
    
    if disease == 'Healthy Control':
        _, _, gait_speed = healthy_gait_intuitive(t)
        ax.plot(t, gait_speed, color=color, linewidth=2.5, label='Gait Speed')
        ax.text(0.1, 135, 'Normal:\\n120±10 steps/min', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
    elif disease == "Parkinson's Disease":
        _, _, gait_speed = parkinsons_gait_intuitive(t)
        ax.plot(t, gait_speed, color=color, linewidth=2.5, label='Gait Speed')
        ax.text(0.1, 105, 'Slow & Variable:\\n80±15 steps/min\\n(Festination)', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        
    elif disease == 'Cerebral Palsy':
        _, _, gait_speed = cerebral_palsy_gait_intuitive(t)
        ax.plot(t, gait_speed, color=color, linewidth=2.5, label='Gait Speed')
        ax.text(0.1, 130, 'Unsteady:\\n90±25 steps/min\\n(Incoordination)', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
    else:  # Stroke
        _, _, _, _, gait_speed = stroke_gait_intuitive(t)
        ax.plot(t, gait_speed, color=color, linewidth=2.5, label='Overall Speed')
        ax.text(0.1, 85, 'Compensatory:\\n70±10 steps/min\\n(Hemiplegia)', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    ax.set_title(f'Gait Frequency', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (gait cycles)', fontsize=11)
    ax.set_ylabel('Frequency (steps/min)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(40, 150)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('/home/ghr/me5412/docs/intuitive_disease_gait_patterns.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 创建对比图：技术版 vs 直观版
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# 上图：技术版（加速度）
t_short = np.linspace(0, 1, 100)
healthy_acc = 9.8 + 0.2 * np.sin(2 * np.pi * t_short)
parkinsons_acc = 9.8 + 0.1 * np.sin(2 * np.pi * t_short) + 0.05 * np.random.randn(100)
cerebral_acc = 9.8 + 0.3 * np.sin(2 * np.pi * t_short) + 0.1 * np.sin(6 * np.pi * t_short)
stroke_acc = 9.8 + 0.25 * np.sin(2 * np.pi * t_short) + 0.1 * np.sin(np.pi * t_short)

ax1.plot(t_short, healthy_acc, 'g-', linewidth=2, label='Healthy')
ax1.plot(t_short, parkinsons_acc, 'orange', linewidth=2, label="Parkinson's")
ax1.plot(t_short, cerebral_acc, 'r-', linewidth=2, label='Cerebral Palsy')
ax1.plot(t_short, stroke_acc, 'm-', linewidth=2, label='Stroke')
ax1.axhline(y=9.8, color='gray', linestyle='--', alpha=0.5, label='Gravity baseline')
ax1.set_title('Technical Version: Vertical Acceleration (for Researchers)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Acceleration (m/s²)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.text(0.02, 10.15, 'Less intuitive but\\nscientifically accurate', 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# 下图：直观版（关节角度）
healthy_knee, _, _ = healthy_gait_intuitive(t_short)
parkinsons_knee, _, _ = parkinsons_gait_intuitive(t_short)
cerebral_knee, _, _ = cerebral_palsy_gait_intuitive(t_short)
stroke_knee_h, stroke_knee_a, _, _, _ = stroke_gait_intuitive(t_short)

ax2.plot(t_short, healthy_knee, 'g-', linewidth=2, label='Healthy')
ax2.plot(t_short, parkinsons_knee, 'orange', linewidth=2, label="Parkinson's")
ax2.plot(t_short, cerebral_knee, 'r-', linewidth=2, label='Cerebral Palsy')
ax2.plot(t_short, stroke_knee_h, 'm-', linewidth=2, label='Stroke (Healthy side)')
ax2.plot(t_short, stroke_knee_a, 'm--', linewidth=2, alpha=0.7, label='Stroke (Affected side)')
ax2.set_title('Intuitive Version: Knee Joint Angle (for Clinicians & Patients)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (gait cycle)', fontsize=12)
ax2.set_ylabel('Knee Angle (degrees)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.text(0.02, 90, 'More intuitive and\\nclinically meaningful', 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.suptitle('Comparison: Technical vs Intuitive Visualization Approaches', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ghr/me5412/docs/technical_vs_intuitive_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 更直观的步态分析图片已生成:")
print("1. intuitive_disease_gait_patterns.png - 关节角度和步态参数版本")
print("2. technical_vs_intuitive_comparison.png - 技术版与直观版对比")
print("\\n💡 建议在PPT中:")
print("- 技术部分使用加速度图（体现科学严谨性）")
print("- 结果展示使用关节角度图（便于理解和解释）")
print("- 可以展示两种版本的对比，说明研究的实用性")
