import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Patch, Rectangle, FancyBboxPatch
class SafetyFieldCalculator:
    """行车安全场计算器 - 大幅提高危险红色范围版本"""

    def __init__(self):
        # 安全场模型参数
        self.K = 0.1  # 场强系数提高，增加基础风险
        self.k_1 = 1.3  # 距离衰减系数降低，让远距离也有风险
        self.A = 0.3  # 基础方向风险提高
        self.B = 1.0  # 方向放大系数提高
        self.C = 2.2  # 方向锐度系数提高
        self.D = 0.15  # 速度影响系数提高
        self.E = 1.8  # 速度非线性因子提高

        # 大幅提高危险范围：显著降低安全阈值
        self.safe_threshold = 0.5  # 原0.7 → 0.5 (大幅降低安全标准)
        self.warning_threshold = 0.28  # 原0.4 → 0.2 (大幅扩大危险范围)

        # 激进权重配置 - 速度主导，更容易触发危险
        self.safety_weights = {
            'distance_safety': 0.25,  # 距离权重大幅降低
            'angle_safety': 0.15,  # 角度权重极低
            'speed_safety': 0.50,  # 速度权重主导 (原0.4 → 0.65)
            'mass_safety': 0.10  # 质量权重保持不变
        }

    def calculate_virtual_mass(self, obj_type, speed_kmh, physical_mass=1400):
        """计算虚拟质量 - 提高风险"""
        # 物体类型系数 - 提高所有类型的风险系数
        type_coefficients = {
            'bicycle': 0.4,  # 自行车风险提高
            'motorcycle': 0.7,  # 摩托车风险大幅提高
            'car': 1.2,  # 轿车风险提高
            'truck': 2.5,  # 卡车风险大幅提高
            'person': 0.3  # 行人风险提高
        }

        T_i = type_coefficients.get(obj_type, 0.8)  # 默认风险提高

        # 速度影响 - 使用更激进的计算
        speed_effect = 2.0e-14 * (speed_kmh ** 7.0) + 0.5  # 提高速度影响

        return physical_mass * T_i * speed_effect

    def calculate_direction_risk(self, angle_degrees):
        """计算方向性风险函数 - 提高所有角度的风险"""
        # 将角度转换为弧度
        theta_b_rad = np.radians(angle_degrees)

        # 方向风险函数 - 提高基础风险
        return self.A + self.B * (abs(np.cos(theta_b_rad / 2)) ** self.C)

    def calculate_safety_factor(self, distance_m, virtual_mass, angle_degrees, closing_speed_kmh, road_condition=1.0):
        """计算安全系数 - 大幅提高危险红色范围"""

        # 1. 基础安全系数 (基于距离) - 提高距离风险
        base_field = (self.K * road_condition * virtual_mass) / (distance_m ** self.k_1)
        SF_base = 1 / (1 + base_field)

        # 2. 方向安全系数 - 提高方向风险
        direction_risk = self.calculate_direction_risk(angle_degrees)
        SF_direction = 1 / direction_risk

        # 3. 速度安全系数 - 使用极度敏感的速度安全计算
        SF_speed = self.calculate_aggressive_speed_safety(closing_speed_kmh)

        # 4. 质量安全系数 - 提高质量风险
        base_mass = 1400
        normalized_mass = min(virtual_mass / base_mass, 4.0)  # 提高质量上限
        SF_mass = 1.0 / (1.0 + 0.8 * normalized_mass)  # 提高质量影响

        # 使用激进权重进行综合计算
        safety_factor = (
                SF_base * self.safety_weights['distance_safety'] +
                SF_direction * self.safety_weights['angle_safety'] +
                SF_speed * self.safety_weights['speed_safety'] +
                SF_mass * self.safety_weights['mass_safety']
        )

        # 确保安全系数在合理范围内
        safety_factor = max(0.0, min(1.0, safety_factor))

        return safety_factor, SF_base, SF_direction, SF_speed, SF_mass

    def calculate_aggressive_speed_safety(self, closing_speed_kmh):
        """激进的速度安全计算 - 极度敏感，极易触发危险"""
        if closing_speed_kmh <= 0:
            return 0.9  # 即使静止也有一定风险

        # 极度敏感的分段函数，极容易触发危险状态
        if closing_speed_kmh < 3:
            # 极低速阶段：快速下降
            speed_safety = 0.9 - (closing_speed_kmh / 3) * 0.5
        elif closing_speed_kmh < 10:
            # 低速阶段：急剧下降
            speed_safety = 0.4 - ((closing_speed_kmh - 3) / 7) * 0.35
        elif closing_speed_kmh < 25:
            # 中速阶段：极低安全系数
            speed_safety = 0.05 - ((closing_speed_kmh - 10) / 15) * 0.04
        else:
            # 高速：几乎为零的安全系数
            speed_safety = 0.01

        return max(0.0, min(1.0, speed_safety))

    def get_safety_level(self, safety_factor):
        """根据安全系数获取安全等级 - 大幅提高红色范围"""
        # 现在只有很高的安全系数才会显示绿色
        if safety_factor >= self.safe_threshold:
            return "安全", (0, 255, 0)  # 绿色 - 只有很少情况
        elif safety_factor >= self.warning_threshold:
            return "警戒", (0, 255, 255)  # 黄色 - 中等范围
        else:
            return "危险", (0, 0, 255)  # 红色 - 大部分情况

    def print_safety_weights(self):
        """打印当前使用的安全权重"""
        print("📊 当前安全系数权重配置 (大幅提高危险范围):")
        print(f"  距离安全: {self.safety_weights['distance_safety']:.1%}")
        print(f"  角度安全: {self.safety_weights['angle_safety']:.1%}")
        print(f"  速度安全: {self.safety_weights['speed_safety']:.1%}")
        print(f"  质量安全: {self.safety_weights['mass_safety']:.1%}")
        print(f"  总权重: {sum(self.safety_weights.values()):.0%}")
        print(f"  安全阈值: {self.safe_threshold} (原0.7)")
        print(f"  警戒阈值: {self.warning_threshold} (原0.4)")
        print("⚠️  危险模式: 大部分情况将显示红色警告")



class SafetyFieldHeatmapVisualizerWithAnnotations:
    """行车安全场热点图可视化（带中文批注版）"""

    def __init__(self, calculator):
        self.calculator = calculator

    def create_safety_heatmap_with_annotations(self, ego_position=(0, 0), target_speed=40.0,
                                               target_angle=45.0, target_type='car',
                                               save_path=None):
        """
        创建带中文批注的安全场热点图
        """

        # 创建图形，设置白色背景
        fig = plt.figure(figsize=(18, 14), facecolor='white')

        # ==================== 创建网格布局 ====================
        # 使用GridSpec创建更灵活的布局
        gs = fig.add_gridspec(12, 12, hspace=0.5, wspace=0.5)

        # 主热点图区域
        ax_main = fig.add_subplot(gs[0:7, 0:7])

        # 右侧信息面板
        ax_info = fig.add_subplot(gs[0:4, 8:12])
        ax_info.axis('off')

        # 底部分析图区域
        ax_angle = fig.add_subplot(gs[8:11, 0:5])
        ax_distance = fig.add_subplot(gs[8:11, 6:11])

        # 颜色条区域
        ax_colorbar = fig.add_subplot(gs[0:7, 7])

        # ==================== 主热点图生成 ====================
        # 创建计算网格
        grid_size = 101
        x_range = 50
        x = np.linspace(-x_range, x_range, grid_size)
        y = np.linspace(-x_range, x_range, grid_size)
        X, Y = np.meshgrid(x, y)

        # 计算安全系数网格
        safety_grid = np.zeros_like(X)
        ego_x, ego_y = ego_position

        for i in range(grid_size):
            for j in range(grid_size):
                dx = X[i, j] - ego_x
                dy = Y[i, j] - ego_y
                distance = np.sqrt(dx ** 2 + dy ** 2)

                if distance < 0.5:
                    safety_grid[i, j] = 1.0
                    continue

                angle = np.degrees(np.arctan2(dy, dx))
                relative_angle = angle - target_angle

                while relative_angle > 180:
                    relative_angle -= 360
                while relative_angle < -180:
                    relative_angle += 360

                virtual_mass = self.calculator.calculate_virtual_mass(
                    target_type, target_speed, 1400
                )

                safety_factor, _, _, _, _ = self.calculator.calculate_safety_factor(
                    distance, virtual_mass, relative_angle, target_speed, 1.0
                )

                safety_grid[i, j] = safety_factor

        # 创建颜色映射
        colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
        cmap = mcolors.LinearSegmentedColormap.from_list("safety_cmap", colors, N=256)

        # 绘制热点图
        im = ax_main.imshow(safety_grid, extent=[-x_range, x_range, -x_range, x_range],
                            cmap=cmap, aspect='auto', alpha=0.85, origin='lower')

        # ==================== 添加主要中文批注 ====================

        # 1. 标题和整体说明
        fig.suptitle('行车安全场模型综合分析图\n(基于多普勒效应原理的危险预警系统)',
                     fontsize=18, fontweight='bold', fontname='SimHei', y=0.98)

        # 主图标题
        ax_main.set_title('安全系数空间分布热点图', fontsize=14, fontweight='bold',
                          fontname='SimHei', pad=15, color='darkblue')

        # 2. 自车位置批注
        ax_main.annotate('🚗 自车位置\n(安全场中心)',
                         xy=(ego_x, ego_y), xytext=(-40, 0),
                         arrowprops=dict(arrowstyle='->', color='blue', lw=2,
                                         connectionstyle="arc3,rad=0.3"),
                         fontsize=10, fontname='SimHei',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        # 3. 目标物体批注
        target_distance = 25
        target_x = ego_x + target_distance * np.cos(np.radians(target_angle))
        target_y = ego_y + target_distance * np.sin(np.radians(target_angle))

        ax_main.scatter(target_x, target_y, s=250, c='red', marker='s',
                        edgecolors='black', linewidth=2, zorder=10)

        ax_main.annotate(f'🚙 目标车辆\n类型: {target_type}\n速度: {target_speed}km/h\n方向: {target_angle}°',
                         xy=(target_x, target_y), xytext=(target_x + 10, target_y + 10),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2),
                         fontsize=10, fontname='SimHei',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

        # 4. 梯度方向批注（模仿图2-12）
        ax_main.annotate('📈 梯度方向\n(风险增加最快方向)\n类比多普勒效应中的波前密集区',
                         xy=(10, 10), xytext=(25, 25),
                         arrowprops=dict(arrowstyle='->', color='darkred', lw=2,
                                         connectionstyle="arc3,rad=0.2"),
                         fontsize=10, fontname='SimHei',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        # 5. 等高线批注
        contours = ax_main.contour(X, Y, safety_grid, levels=[0.28, 0.5],
                                   colors=['white', 'white'],
                                   linewidths=[1.5, 2], linestyles=['--', '-'])
        ax_main.clabel(contours, inline=True, fontsize=9, fmt='%.2f')

        ax_main.annotate('📊 安全等级边界\n虚线: 警戒阈值(0.28)\n实线: 安全阈值(0.50)',
                         xy=(30, 30), xytext=(35, 40),
                         arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                         fontsize=9, fontname='SimHei', color='white',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

        # 6. 区域划分批注
        # 危险区域
        ax_main.annotate('🔴 高危险区\n(紧急制动建议)',
                         xy=(15, -20), xytext=(25, -35),
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                         fontsize=9, fontname='SimHei',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))

        # 警戒区域
        ax_main.annotate('🟡 警戒区域\n(谨慎驾驶)',
                         xy=(-15, 30), xytext=(-35, 35),
                         arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                         fontsize=9, fontname='SimHei',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # 安全区域
        ax_main.annotate('🟢 相对安全区\n(保持车距)',
                         xy=(-30, -30), xytext=(-45, -40),
                         arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                         fontsize=9, fontname='SimHei',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # 设置主图属性
        ax_main.set_xlabel('横向距离 (米) →', fontsize=12, fontweight='bold', fontname='SimHei')
        ax_main.set_ylabel('纵向距离 (米) →', fontsize=12, fontweight='bold', fontname='SimHei')
        ax_main.grid(True, alpha=0.2, linestyle='--')
        ax_main.set_aspect('equal')

        # ==================== 右侧信息面板 ====================
        # 添加标题
        ax_info.text(0.1, 0.95, '📋 模型参数与说明', fontsize=14, fontweight='bold',
                     fontname='SimHei', transform=ax_info.transAxes)

        # 参数说明
        param_text = [
            "安全场模型核心参数:",
            f"• 场强系数 K = {self.calculator.K}",
            f"• 距离衰减 k₁ = {self.calculator.k_1}",
            f"• 方向参数 A,B,C = {self.calculator.A}, {self.calculator.B}, {self.calculator.C}",
            f"• 速度参数 D,E = {self.calculator.D}, {self.calculator.E}",
            "",
            "安全阈值设置:",
            f"• 警戒阈值 = {self.calculator.warning_threshold}",
            f"• 安全阈值 = {self.calculator.safe_threshold}",
            "",
            "权重配置 (危险模式):",
            f"• 距离安全: {self.calculator.safety_weights['distance_safety']:.0%}",
            f"• 角度安全: {self.calculator.safety_weights['angle_safety']:.0%}",
            f"• 速度安全: {self.calculator.safety_weights['speed_safety']:.0%}",
            f"• 质量安全: {self.calculator.safety_weights['mass_safety']:.0%}",
            "",
            "⚠️ 当前为危险预警模式",
            "   红色区域大幅扩大，安全标准提高"
        ]

        for i, line in enumerate(param_text):
            y_pos = 0.85 - i * 0.045
            color = 'red' if '危险' in line else 'black'
            weight = 'bold' if '警戒' in line or '安全' in line else 'normal'
            ax_info.text(0.1, y_pos, line, fontsize=9, fontname='SimHei',
                         transform=ax_info.transAxes, color=color, fontweight=weight)

        # 添加物理原理说明
        physics_text = [
            "📚 物理原理参考:",
            "• 基于多普勒效应: f = f₀(1 + v·cosθ/c)",
            "• 类比波传播: 前方风险>后方风险",
            "• 动能场分布: E_v ∝ M·R·k₃·cosθ",
            "• 梯度方向: grad(E_v)/|grad(E_v)|",
            "",
            "📍 坐标变换公式:",
            "[X]   [cosφ  sinφ] [x]",
            "[Y] = [-sinφ cosφ] [y]",
            "(旋转φ角度到速度方向)"
        ]

        for i, line in enumerate(physics_text):
            y_pos = 0.30 - i * 0.04
            ax_info.text(0.1, y_pos, line, fontsize=8, fontname='SimHei',
                         transform=ax_info.transAxes, fontfamily='monospace')

        # ==================== 颜色条 ====================
        cbar = plt.colorbar(im, cax=ax_colorbar, orientation='vertical')
        cbar.set_label('安全系数 (0=危险, 1=安全)', fontsize=10, fontname='SimHei', fontweight='bold')

        # 标记阈值
        cbar.ax.axhline(y=0.28, color='white', linestyle='--', linewidth=2)
        cbar.ax.axhline(y=0.50, color='white', linestyle='-', linewidth=2)

        cbar.ax.text(1.5, 0.28, '警戒线', transform=cbar.ax.transAxes,
                     ha='left', va='center', fontsize=9, fontname='SimHei',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
        cbar.ax.text(1.5, 0.50, '安全线', transform=cbar.ax.transAxes,
                     ha='left', va='center', fontsize=9, fontname='SimHei',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="green", alpha=0.7))

        # ==================== 角度分析图 ====================
        # 计算角度变化数据
        fixed_distance = 20
        angles = np.linspace(-180, 180, 361)
        virtual_mass = self.calculator.calculate_virtual_mass(target_type, target_speed, 1400)
        safety_by_angle = []

        for angle in angles:
            safety_factor, _, _, _, _ = self.calculator.calculate_safety_factor(
                fixed_distance, virtual_mass, angle, target_speed, 1.0
            )
            safety_by_angle.append(safety_factor)

        # 绘制角度图
        ax_angle.plot(angles, safety_by_angle, 'b-', linewidth=2)
        ax_angle.fill_between(angles, 0, safety_by_angle, alpha=0.3, color='blue')

        # 关键角度标注
        key_angles = [-180, -90, 0, 90, 180]
        for angle in key_angles:
            idx = np.argmin(np.abs(angles - angle))
            safety_value = safety_by_angle[idx]
            color = 'red' if safety_value < 0.28 else ('yellow' if safety_value < 0.5 else 'green')
            ax_angle.plot(angle, safety_value, 'o', color=color, markersize=8)
            direction_text = {0: '正前方', 90: '正左侧', -90: '正右侧', 180: '正后方'}.get(angle, f'{angle}°')
            ax_angle.text(angle, safety_value + 0.07, direction_text,
                          ha='center', va='bottom', fontsize=9, fontname='SimHei')

        # 设置角度图属性
        ax_angle.set_xlabel('相对角度 (度)', fontsize=11, fontname='SimHei')
        ax_angle.set_ylabel('安全系数', fontsize=11, fontname='SimHei')
        ax_angle.set_title('📐 安全系数随角度变化分析\n(距离固定: 20米)',
                           fontsize=12, fontname='SimHei', fontweight='bold')
        ax_angle.grid(True, alpha=0.3)
        ax_angle.set_xlim(-180, 180)
        ax_angle.set_ylim(0, 1)
        ax_angle.axhline(y=0.28, color='red', linestyle='--', alpha=0.5, label='警戒线')
        ax_angle.axhline(y=0.50, color='green', linestyle='--', alpha=0.5, label='安全线')
        ax_angle.legend(loc='upper right', fontsize=9, prop={'family': 'SimHei'})

        # 角度图批注
        ax_angle.annotate('✅ 正前方相对安全', xy=(0, safety_by_angle[180]), xytext=(30, 0.8),
                          arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                          fontsize=9, fontname='SimHei',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        ax_angle.annotate('⚠️ 侧向风险较高', xy=(90, safety_by_angle[270]), xytext=(120, 0.3),
                          arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                          fontsize=9, fontname='SimHei',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

        # ==================== 距离分析图 ====================
        # 计算距离变化数据
        fixed_angle = 0
        distances = np.linspace(1, 50, 50)
        safety_by_distance = []

        for distance in distances:
            safety_factor, _, _, _, _ = self.calculator.calculate_safety_factor(
                distance, virtual_mass, fixed_angle, target_speed, 1.0
            )
            safety_by_distance.append(safety_factor)

        # 绘制距离图
        ax_distance.plot(distances, safety_by_distance, 'r-', linewidth=2)
        ax_distance.fill_between(distances, 0, safety_by_distance, alpha=0.3, color='red')

        # 关键距离标注
        key_distances = [5, 10, 20, 30, 40]
        for distance in key_distances:
            idx = np.argmin(np.abs(distances - distance))
            safety_value = safety_by_distance[idx]
            color = 'red' if safety_value < 0.28 else ('yellow' if safety_value < 0.5 else 'green')
            ax_distance.plot(distance, safety_value, 'o', color=color, markersize=8)

            # 添加安全等级标签
            if distance <= 10:
                level = "⚠️ 紧急距离"
            elif distance <= 20:
                level = "🟡 警戒距离"
            else:
                level = "✅ 安全距离"

            ax_distance.text(distance, safety_value + 0.07, f'{distance}m\n{level}',
                             ha='center', va='bottom', fontsize=8, fontname='SimHei')

        # 设置距离图属性
        ax_distance.set_xlabel('距离 (米)', fontsize=11, fontname='SimHei')
        ax_distance.set_ylabel('安全系数', fontsize=11, fontname='SimHei')
        ax_distance.set_title('📏 安全系数随距离变化分析\n(角度固定: 正前方0°)',
                              fontsize=12, fontname='SimHei', fontweight='bold')
        ax_distance.grid(True, alpha=0.3)
        ax_distance.set_xlim(0, 50)
        ax_distance.set_ylim(0, 1)
        ax_distance.axhline(y=0.28, color='red', linestyle='--', alpha=0.5)
        ax_distance.axhline(y=0.50, color='green', linestyle='--', alpha=0.5)

        # 距离图批注
        ax_distance.annotate('🚨 近距离高危险\n(建议保持>20米)',
                             xy=(10, safety_by_distance[9]), xytext=(15, 0.2),
                             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                             fontsize=9, fontname='SimHei',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

        ax_distance.annotate('✅ 远距离相对安全\n(保持当前状态)',
                             xy=(40, safety_by_distance[39]), xytext=(30, 0.6),
                             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                             fontsize=9, fontname='SimHei',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # ==================== 底部统计信息 ====================
        # 计算统计数据
        stats = {
            'danger': np.sum(safety_grid < 0.28) / safety_grid.size * 100,
            'warning': np.sum((safety_grid >= 0.28) & (safety_grid < 0.5)) / safety_grid.size * 100,
            'safe': np.sum(safety_grid >= 0.5) / safety_grid.size * 100
        }

        # 添加统计信息框
        stats_text = f"""
        📊 区域统计（100×100网格）:

        高危险区: {stats['danger']:.1f}%  (红色区域)
        警戒区域: {stats['warning']:.1f}%  (黄色区域)
        相对安全: {stats['safe']:.1f}%    (绿色区域)

        安全系数范围: {np.min(safety_grid):.3f} ~ {np.max(safety_grid):.3f}
        平均安全系数: {np.mean(safety_grid):.3f}

        💡 安全建议:
        • 红色区域: 紧急制动或避让
        • 黄色区域: 减速并提高警惕
        • 绿色区域: 保持安全距离行驶
        """

        # 在图形底部添加统计信息
        fig.text(0.02, 0.02, stats_text, fontsize=9, fontname='SimHei',
                 verticalalignment='bottom',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        # 添加制作信息
        fig.text(0.98, 0.02, '© 行车安全场分析系统\n基于多普勒效应危险预警模型',
                 fontsize=8, fontname='SimHei', ha='right',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

        plt.tight_layout(rect=[0, 0.08, 1, 0.98])

        # 保存图像
        if save_path:
            try:
                import os
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✅ 带批注的热点图已保存到: {save_path}")
            except Exception as e:
                print(f"⚠️ 保存图像时出错: {e}")
                plt.savefig("safety_field_with_annotations.png", dpi=300, bbox_inches='tight', facecolor='white')
                print("✅ 图像已保存到当前目录: safety_field_with_annotations.png")

        plt.show()

        return fig, stats


# 主程序
if __name__ == "__main__":
    # 创建安全场计算器
    calculator = SafetyFieldCalculator()

    # 打印参数配置
    calculator.print_safety_weights()

    # 创建可视化器
    visualizer = SafetyFieldHeatmapVisualizerWithAnnotations(calculator)

    print("\n" + "=" * 60)
    print("🚗 行车安全场带批注热点图生成器")
    print("=" * 60)

    # 生成带中文批注的热点图
    fig, stats = visualizer.create_safety_heatmap_with_annotations(
        ego_position=(0, 0),
        target_speed=40.0,
        target_angle=45.0,
        target_type='car',
        save_path="安全场分析_带批注.png"
    )

    print(f"\n📈 分析完成！危险区域占比: {stats['danger']:.1f}%")
    print("💡 查看生成图像获取详细安全分析")