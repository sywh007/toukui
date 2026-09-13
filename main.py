import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import time
import json
import random
import numpy as np
from collections import defaultdict


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理 - 使用更平衡的数据增强
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.4),  # 适当增加翻转概率
    transforms.RandomRotation(8),  # 减小旋转角度
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class VehicleAngleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.angle_labels = []

        self.angle_mapping = {
            '0': 0, '5': 1, '10': 2, '15': 3,
            '20': 4, '25': 5, '30': 6
        }

        self._scan_data_directory()
        print(f"图片数量: {len(self.image_paths)}")

    def _scan_data_directory(self):
        # 扫描所有角度目录，不考虑车头车尾
        for angle_dir in ['0', '5', '10', '15', '20', '25', '30']:
            angle_path = os.path.join(self.data_dir, angle_dir)
            if not os.path.exists(angle_path):
                continue

            for img_name in os.listdir(angle_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(angle_path, img_name)
                    self.image_paths.append(img_path)
                    self.angle_labels.append(self.angle_mapping[angle_dir])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        angle_label = self.angle_labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (64, 64), color='black')

        if self.transform:
            image = self.transform(image)

        return image, angle_label


class AngleCNN(nn.Module):
    def __init__(self, conv_layers=3, start_channels=32, pool_kernel=2, pool_stride=2,
                 dropout_rate=0.3, use_batch_norm=True, input_size=64):
        super().__init__()

        self.conv_layers = conv_layers
        self.start_channels = start_channels
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # 构建卷积层
        self.conv_blocks = nn.ModuleList()
        current_channels = 3

        # 更平衡的通道增长策略
        if conv_layers == 2:
            channel_growth = [start_channels, start_channels * 2]
        elif conv_layers == 3:
            channel_growth = [start_channels, start_channels * 2, start_channels * 3]
        else:  # 4层
            channel_growth = [start_channels, start_channels * 2, start_channels * 3, start_channels * 3]

        for i in range(conv_layers):
            out_channels = channel_growth[i]

            layers = []
            # 第一个卷积层
            layers.append(nn.Conv2d(current_channels, out_channels, 3, padding=1, bias=False))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # 第二个卷积层
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            # 池化层
            layers.append(nn.MaxPool2d(pool_kernel, pool_stride))

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate / 3))

            self.conv_blocks.append(nn.Sequential(*layers))
            current_channels = out_channels

        # 全局特征聚合
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = current_channels

        # 角度分类器
        self.angle_classifier = self._create_classifier(7)

        # 权重初始化
        self._initialize_weights()

    def _create_classifier(self, num_classes):
        """创建角度分类器"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.8),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.6),
            nn.Linear(48, num_classes)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.global_avg_pool(x)
        features = x.view(x.size(0), -1)

        angle_output = self.angle_classifier(features)

        return angle_output


def evaluate_model(model, testloader, criterion, device):
    model.eval()
    angle_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, angle_labels in testloader:
            inputs = inputs.to(device)
            angle_labels = angle_labels.to(device)

            angle_output = model(inputs)
            loss = criterion(angle_output, angle_labels)
            total_loss += loss.item()

            _, angle_predicted = torch.max(angle_output, 1)
            angle_correct += (angle_predicted == angle_labels).sum().item()
            total_samples += angle_labels.size(0)

    angle_accuracy = 100 * angle_correct / total_samples
    avg_loss = total_loss / len(testloader)

    return angle_accuracy, avg_loss


def train_model(config, trainloader, testloader, epochs=40):
    print(f"训练: {config['description']}")

    try:
        set_seed(42)

        model = AngleCNN(
            conv_layers=config['conv_layers'],
            start_channels=config['start_channels'],
            pool_kernel=config['pool_kernel'],
            pool_stride=config['pool_stride'],
            dropout_rate=config.get('dropout_rate', 0.3),
            use_batch_norm=config.get('use_batch_norm', True),
            input_size=64
        )
        model = model.to(device)

        # 根据模型复杂度调整学习率
        base_lr = 0.01
        if config['conv_layers'] >= 4:
            base_lr = 0.008

        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

        # 使用更稳定的优化器配置
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # 使用余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

        model.train()
        best_angle_accuracy = 0
        best_epoch = 0
        patience = 12
        patience_counter = 0
        best_model_state = None

        # 记录训练历史
        train_history = {
            'angle_acc': []
        }

        for epoch in range(epochs):
            running_total_loss = 0.0
            angle_correct = 0
            total_samples = 0

            # 训练阶段
            model.train()
            for inputs, angle_labels in trainloader:
                inputs = inputs.to(device)
                angle_labels = angle_labels.to(device)

                optimizer.zero_grad()
                angle_output = model(inputs)

                loss = criterion(angle_output, angle_labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_total_loss += loss.item()

                _, angle_predicted = torch.max(angle_output, 1)
                angle_correct += (angle_predicted == angle_labels).sum().item()
                total_samples += angle_labels.size(0)

            scheduler.step()

            # 验证阶段
            model.eval()
            val_angle_acc, val_loss = evaluate_model(model, testloader, criterion, device)
            model.train()

            # 记录历史
            train_history['angle_acc'].append(val_angle_acc)

            # 使用平滑的准确率进行早停
            smoothed_accuracy = np.mean(train_history['angle_acc'][-3:]) if len(
                train_history['angle_acc']) >= 3 else val_angle_acc

            if smoothed_accuracy > best_angle_accuracy:
                best_angle_accuracy = smoothed_accuracy
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict().copy(),
                    'val_angle_accuracy': val_angle_acc
                }
            else:
                patience_counter += 1

            # 进度显示
            if (epoch + 1) % 8 == 0 or epoch == 0 or epoch == epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1:2d}: 损失 {running_total_loss / len(trainloader):.3f}, "
                      f"角度准确率 {val_angle_acc:.1f}%")

            # 早停检查
            if patience_counter >= patience:
                print(f"早停: epoch {epoch + 1}")
                break

        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state['model_state_dict'])

        # 最终评估
        final_angle_acc, final_loss = evaluate_model(model, testloader, criterion, device)

        result = {
            'config': config,
            'final_angle_accuracy': final_angle_acc,
            'final_loss': final_loss,
            'best_angle_accuracy': best_angle_accuracy,
            'best_epoch': best_epoch + 1,
            'feature_dim': model.feature_dim,
            'total_epochs': epoch + 1,
            'early_stopped': patience_counter >= patience,
            'status': 'success',
            'train_history': train_history
        }

        print(f"完成: 角度准确率 {final_angle_acc:.1f}%")

        return result, model

    except Exception as e:
        print(f"训练失败")
        return {
                   'config': config,
                   'final_angle_accuracy': 0,
                   'final_loss': float('inf'),
                   'best_angle_accuracy': 0,
                   'status': 'failed',
                   'error': str(e)
               }, None


def get_balanced_configurations():
    """生成平衡的配置组合"""
    configs = []

    # 专注于平衡性能的配置
    balanced_configs = [
        # 中等复杂度 - 平衡选择
        {'conv_layers': 3, 'start_channels': 28, 'pool_kernel': 2, 'pool_stride': 2, 'dropout_rate': 0.35},
        {'conv_layers': 3, 'start_channels': 32, 'pool_kernel': 2, 'pool_stride': 2, 'dropout_rate': 0.4},
        {'conv_layers': 3, 'start_channels': 24, 'pool_kernel': 2, 'pool_stride': 2, 'dropout_rate': 0.3},

        # 稍深网络
        {'conv_layers': 4, 'start_channels': 20, 'pool_kernel': 2, 'pool_stride': 2, 'dropout_rate': 0.4},
        {'conv_layers': 4, 'start_channels': 18, 'pool_kernel': 2, 'pool_stride': 2, 'dropout_rate': 0.35},

        # 简单但稳定的网络
        {'conv_layers': 2, 'start_channels': 36, 'pool_kernel': 2, 'pool_stride': 2, 'dropout_rate': 0.25},
        {'conv_layers': 2, 'start_channels': 40, 'pool_kernel': 2, 'pool_stride': 2, 'dropout_rate': 0.3},

        # 不同池化策略
        {'conv_layers': 3, 'start_channels': 28, 'pool_kernel': 3, 'pool_stride': 2, 'dropout_rate': 0.35},
        {'conv_layers': 3, 'start_channels': 24, 'pool_kernel': 2, 'pool_stride': 1, 'dropout_rate': 0.4},
    ]

    for i, cfg in enumerate(balanced_configs):
        config = {
            'conv_layers': cfg['conv_layers'],
            'start_channels': cfg['start_channels'],
            'pool_kernel': cfg['pool_kernel'],
            'pool_stride': cfg['pool_stride'],
            'dropout_rate': cfg['dropout_rate'],
            'use_batch_norm': True,
            'description': f"{cfg['conv_layers']}层_通道{cfg['start_channels']}_Dropout{cfg['dropout_rate']}"
        }
        configs.append(config)

    print(f"配置数量: {len(configs)}")
    return configs


def analyze_parameter_performance(results):
    """分析各个参数对性能的影响"""
    if not results:
        return

    successful_results = [r for r in results if r.get('status') == 'success']
    if not successful_results:
        return

    print(f"\n" + "=" * 70)
    print("📊 参数性能分析")
    print("=" * 70)

    # 分析卷积层数的影响
    print(f"\n🧩 卷积层数分析:")
    layer_performance = {}
    for result in successful_results:
        layers = result['config']['conv_layers']
        if layers not in layer_performance:
            layer_performance[layers] = []
        layer_performance[layers].append(result['final_angle_accuracy'])

    for layers, scores in sorted(layer_performance.items()):
        avg_score = np.mean(scores)
        max_score = max(scores)
        min_score = min(scores)
        print(f"  {layers}层: 平均 {avg_score:.1f}, 最高 {max_score:.1f}, 最低 {min_score:.1f}, 样本数 {len(scores)}")

    # 分析起始通道数的影响
    print(f"\n🎯 起始通道数分析:")
    channel_performance = {}
    for result in successful_results:
        channels = result['config']['start_channels']
        if channels not in channel_performance:
            channel_performance[channels] = []
        channel_performance[channels].append(result['final_angle_accuracy'])

    for channels, scores in sorted(channel_performance.items()):
        avg_score = np.mean(scores)
        max_score = max(scores)
        min_score = min(scores)
        print(f"  通道{channels}: 平均 {avg_score:.1f}, 最高 {max_score:.1f}, 最低 {min_score:.1f}, 样本数 {len(scores)}")

    # 分析Dropout率的影响
    print(f"\n🛡️ Dropout率分析:")
    dropout_performance = {}
    for result in successful_results:
        dropout = result['config']['dropout_rate']
        if dropout not in dropout_performance:
            dropout_performance[dropout] = []
        dropout_performance[dropout].append(result['final_angle_accuracy'])

    for dropout, scores in sorted(dropout_performance.items()):
        avg_score = np.mean(scores)
        max_score = max(scores)
        min_score = min(scores)
        print(f"  Dropout{dropout}: 平均 {avg_score:.1f}, 最高 {max_score:.1f}, 最低 {min_score:.1f}, 样本数 {len(scores)}")

    # 分析池化策略的影响
    print(f"\n📏 池化策略分析:")
    pool_performance = {}
    for result in successful_results:
        pool_key = f"{result['config']['pool_kernel']}x{result['config']['pool_stride']}"
        if pool_key not in pool_performance:
            pool_performance[pool_key] = []
        pool_performance[pool_key].append(result['final_angle_accuracy'])

    for pool, scores in sorted(pool_performance.items()):
        avg_score = np.mean(scores)
        max_score = max(scores)
        min_score = min(scores)
        print(f"  池化{pool}: 平均 {avg_score:.1f}, 最高 {max_score:.1f}, 最低 {min_score:.1f}, 样本数 {len(scores)}")

    # 分析最佳参数组合
    print(f"\n🏆 最佳参数组合分析:")

    # 按层数分组的最佳配置
    best_by_layers = {}
    for result in successful_results:
        layers = result['config']['conv_layers']
        if layers not in best_by_layers or result['final_angle_accuracy'] > best_by_layers[layers]['final_angle_accuracy']:
            best_by_layers[layers] = result

    for layers, result in sorted(best_by_layers.items()):
        config = result['config']
        print(f"  {layers}层最佳: 通道{config['start_channels']}, Dropout{config['dropout_rate']}, "
              f"池化{config['pool_kernel']}x{config['pool_stride']}, "
              f"角度准确率 {result['final_angle_accuracy']:.1f}%")


def save_model_as_pt(best_model, best_result, filename="best_angle_model.pt"):
    """将最佳模型保存为.pt格式"""
    # 准备模型保存数据
    model_data = {
        # 模型状态字典
        'model_state_dict': best_model.state_dict(),

        # 模型配置
        'model_config': {
            'conv_layers': best_result['config']['conv_layers'],
            'start_channels': best_result['config']['start_channels'],
            'pool_kernel': best_result['config']['pool_kernel'],
            'pool_stride': best_result['config']['pool_stride'],
            'dropout_rate': best_result['config']['dropout_rate'],
            'use_batch_norm': best_result['config']['use_batch_norm'],
            'input_size': 64,
            'feature_dim': best_result['feature_dim']
        },

        # 性能指标
        'performance': {
            'angle_accuracy': best_result['final_angle_accuracy'],
            'best_epoch': best_result['best_epoch'],
            'total_epochs': best_result['total_epochs']
        },

        # 标签映射
        'label_mappings': {
            'angle': {
                '0': '0°',
                '1': '5°',
                '2': '10°',
                '3': '15°',
                '4': '20°',
                '5': '25°',
                '6': '30°'
            }
        },

        # 模型类别信息
        'model_class': 'AngleCNN',
        'model_type': 'angle_classification',

        # 预处理信息
        'preprocessing': {
            'input_size': (64, 64),
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },

        # 时间戳
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'version': '1.0'
    }

    # 保存为.pt文件
    torch.save(model_data, filename)
    print(f"✅ 模型已保存为: {filename}")

    # 打印保存信息
    print(f"\n💾 模型保存详情:")
    print(f"  • 文件格式: .pt (PyTorch Tensor)")
    print(f"  • 文件大小: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
    print(f"  • 包含内容:")
    print(f"    - 模型权重 (model_state_dict)")
    print(f"    - 模型配置 (model_config)")
    print(f"    - 性能指标 (performance)")
    print(f"    - 标签映射 (label_mappings)")
    print(f"    - 预处理信息 (preprocessing)")
    print(f"    - 元数据 (timestamp, version)")

    return filename


def print_best_model_details(best_result, best_model):
    """打印最佳模型的详细参数和标签信息"""
    print(f"\n" + "=" * 80)
    print("🏆 最佳模型详细参数和标签")
    print("=" * 80)

    # 模型配置信息
    best_config = best_result['config']
    print(f"\n📋 模型配置:")
    print(f"  • 卷积层数: {best_config['conv_layers']}")
    print(f"  • 起始通道: {best_config['start_channels']}")
    print(f"  • 池化核: {best_config['pool_kernel']}")
    print(f"  • 池化步长: {best_config['pool_stride']}")
    print(f"  • Dropout率: {best_config['dropout_rate']}")
    print(f"  • 使用批归一化: {best_config['use_batch_norm']}")

    # 性能信息
    print(f"\n🎯 性能指标:")
    print(f"  • 角度准确率: {best_result['final_angle_accuracy']:.2f}%")
    print(f"  • 特征维度: {best_result['feature_dim']}")
    print(f"  • 最佳训练轮次: {best_result['best_epoch']}")

    # 标签映射
    print(f"\n🏷️ 角度标签映射:")
    print(f"  • 0: 0°")
    print(f"  • 1: 5°")
    print(f"  • 2: 10°")
    print(f"  • 3: 15°")
    print(f"  • 4: 20°")
    print(f"  • 5: 25°")
    print(f"  • 6: 30°")

    # 模型层信息
    print(f"\n🧩 模型层结构:")
    print(f"  • 卷积块数量: {len(best_model.conv_blocks)}")

    # 卷积层详细信息
    for i, conv_block in enumerate(best_model.conv_blocks):
        print(f"  • 卷积块 {i + 1}:")
        for j, layer in enumerate(conv_block):
            if isinstance(layer, nn.Conv2d):
                print(f"      - Conv2d-{j}: {layer.in_channels} → {layer.out_channels} (3x3)")
            elif isinstance(layer, nn.BatchNorm2d):
                print(f"      - BatchNorm2d-{j}: {layer.num_features} features")
            elif isinstance(layer, nn.MaxPool2d):
                print(f"      - MaxPool2d-{j}: kernel={layer.kernel_size}, stride={layer.stride}")
            elif isinstance(layer, nn.Dropout2d):
                print(f"      - Dropout2d-{j}: p={layer.p}")

    # 分类器信息
    print(f"\n🔮 角度分类器结构:")
    for i, layer in enumerate(best_model.angle_classifier):
        if isinstance(layer, nn.Linear):
            print(f"      - Linear-{i}: {layer.in_features} → {layer.out_features}")
        elif isinstance(layer, nn.BatchNorm1d):
            print(f"      - BatchNorm1d-{i}: {layer.num_features} features")
        elif isinstance(layer, nn.Dropout):
            print(f"      - Dropout-{i}: p={layer.p}")

    # 参数统计
    total_params = sum(p.numel() for p in best_model.parameters())
    trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    print(f"\n📊 参数统计:")
    print(f"  • 总参数: {total_params:,}")
    print(f"  • 可训练参数: {trainable_params:,}")
    print(f"  • 参数大小: {total_params * 4 / (1024 ** 2):.2f} MB (FP32)")


def hyperparameter_search():
    custom_root = r"E:\workkkkkkk\2222\cnn1\newphoto\back"
    dataset = VehicleAngleDataset(custom_root, transform=transform)

    if len(dataset) == 0:
        print("错误: 没有图片数据")
        return None, [], []

    # 数据集分割
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset.dataset.transform = val_transform

    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"数据: 训练{len(train_dataset)}, 测试{len(test_dataset)}")

    # 获取配置
    configs = get_balanced_configurations()

    # 存储结果
    results = []
    best_result = None
    best_model = None
    best_angle_accuracy = 0

    print(f"搜索 {len(configs)} 种配置")

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] {config['description']}")

        start_time = time.time()
        result, model = train_model(config, trainloader, testloader, epochs=40)
        end_time = time.time()

        if result['status'] == 'success':
            result['training_time'] = end_time - start_time
            results.append(result)

            # 使用角度准确率选择最佳模型
            if result['final_angle_accuracy'] > best_angle_accuracy:
                best_angle_accuracy = result['final_angle_accuracy']
                best_result = result
                best_model = model

            print(f"时间: {result['training_time']:.1f}秒")
        else:
            print(f"失败")

    return best_result, best_model, results


if __name__ == '__main__':
    print("开始角度识别模型搜索")
    start_time = time.time()

    best_result, best_model, all_results = hyperparameter_search()

    end_time = time.time()
    total_time = end_time - start_time

    # 输出结果
    if best_result:
        print(f"\n" + "=" * 60)
        print("🎯 最佳配置")
        print("=" * 60)
        best_config = best_result['config']
        print(f"架构: {best_config['conv_layers']}层卷积")
        print(f"通道: {best_config['start_channels']}")
        print(f"池化: {best_config['pool_kernel']}x{best_config['pool_stride']}")
        print(f"Dropout: {best_config['dropout_rate']}")
        print(f"特征维度: {best_result['feature_dim']}")
        print(f"最佳训练轮次: {best_result['best_epoch']}")
        print(f"角度准确率: {best_result['final_angle_accuracy']:.2f}%")

        # 保存为.pt格式
        pt_filename = save_model_as_pt(best_model, best_result, "best_angle_model.pt")

        # 打印最佳模型的详细参数和标签
        print_best_model_details(best_result, best_model)

    # 显示前5名结果 - 按角度准确率排序
    successful_results = [r for r in all_results if r.get('status') == 'success']
    if successful_results:
        successful_results.sort(key=lambda x: x['final_angle_accuracy'], reverse=True)
        print(f"\n🏆 前5名模型:")
        for i, result in enumerate(successful_results[:5]):
            config = result['config']
            print(f"{i + 1}. {config['description']}")
            print(f"   角度准确率: {result['final_angle_accuracy']:.1f}%")

    # 参数性能分析
    analyze_parameter_performance(all_results)

    print(f"\n⏱️ 总时间: {total_time / 60:.1f}分钟")