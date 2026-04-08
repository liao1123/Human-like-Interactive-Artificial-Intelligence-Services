#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sample.py - 从编译完成的角色卡中按测试点采样

每个测试点（CxDx组合）采样指定数量的角色卡，
用于后续的AI心理对话测试。
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_jsonl(filepath: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """保存为JSONL格式"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def group_by_test_point(cards: List[Dict]) -> Dict[str, List[Dict]]:
    """按测试点分组（CxDx组合）"""
    groups = defaultdict(list)
    
    for card in cards:
        # 只处理编译成功的卡片
        if not card.get('compile_success'):
            continue
        
        category_id = card.get('category_id', '')
        dimension_id = card.get('dimension_id', '')
        
        if category_id and dimension_id:
            key = f"{category_id}{dimension_id}"
            groups[key].append(card)
    
    return dict(groups)


def sample_cards(
    groups: Dict[str, List[Dict]],
    sample_size: int = 100,
    seed: int = None
) -> Dict[str, List[Dict]]:
    """从每个测试点采样指定数量的角色卡"""
    if seed is not None:
        random.seed(seed)
    
    sampled = {}
    
    for key, cards in groups.items():
        if len(cards) <= sample_size:
            # 数量不足，全部保留
            sampled[key] = cards.copy()
        else:
            # 随机采样
            sampled[key] = random.sample(cards, sample_size)
    
    return sampled


def print_stats(
    original_groups: Dict[str, List[Dict]],
    sampled_groups: Dict[str, List[Dict]],
    sample_size: int
):
    """打印采样统计信息"""
    print("\n" + "=" * 70)
    print("采样统计")
    print("=" * 70)
    
    print(f"\n{'测试点':<10} {'原始数量':>10} {'采样数量':>10} {'采样率':>10}")
    print("-" * 45)
    
    total_original = 0
    total_sampled = 0
    insufficient = []
    
    for key in sorted(original_groups.keys()):
        original_count = len(original_groups[key])
        sampled_count = len(sampled_groups.get(key, []))
        rate = sampled_count / original_count * 100 if original_count > 0 else 0
        
        total_original += original_count
        total_sampled += sampled_count
        
        status = "" if original_count >= sample_size else " ⚠️"
        if original_count < sample_size:
            insufficient.append((key, original_count))
        
        print(f"{key:<10} {original_count:>10} {sampled_count:>10} {rate:>9.1f}%{status}")
    
    print("-" * 45)
    overall_rate = total_sampled / total_original * 100 if total_original > 0 else 0
    print(f"{'合计':<10} {total_original:>10} {total_sampled:>10} {overall_rate:>9.1f}%")
    
    print(f"\n测试点数量: {len(original_groups)}")
    print(f"目标采样数: 每测试点 {sample_size} 个")
    
    if insufficient:
        print(f"\n⚠️ 以下测试点数量不足 {sample_size}:")
        for key, count in insufficient:
            print(f"   {key}: 仅有 {count} 个")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='角色卡采样工具')
    parser.add_argument('--input', type=str, default='./role-card/constructed/compiledcards.jsonl',
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default='./role-card/constructed/sampledcards.jsonl',
                        help='输出文件路径')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='每个测试点的采样数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（用于可复现性）')
    parser.add_argument('--stats-only', action='store_true',
                        help='仅显示统计信息，不执行采样')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.input}")
    cards = load_jsonl(args.input)
    print(f"共加载 {len(cards)} 条角色卡")
    
    # 按测试点分组
    groups = group_by_test_point(cards)
    print(f"编译成功的卡片按 {len(groups)} 个测试点分组")
    
    # 采样
    sampled_groups = sample_cards(groups, args.sample_size, args.seed)
    
    # 打印统计
    print_stats(groups, sampled_groups, args.sample_size)
    
    if args.stats_only:
        return
    
    # 合并采样结果
    sampled_cards = []
    for key in sorted(sampled_groups.keys()):
        sampled_cards.extend(sampled_groups[key])
    
    # 保存结果
    print(f"\n保存采样结果: {args.output}")
    save_jsonl(sampled_cards, args.output)
    print(f"共保存 {len(sampled_cards)} 条角色卡")
    
    # 保存统计信息
    stats = {
        'sample_size': args.sample_size,
        'seed': args.seed,
        'total_sampled': len(sampled_cards),
        'test_points': {
            key: {
                'original': len(groups[key]),
                'sampled': len(sampled_groups.get(key, []))
            }
            for key in groups.keys()
        }
    }
    
    stats_path = args.output.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计信息已保存: {stats_path}")


if __name__ == '__main__':
    main()