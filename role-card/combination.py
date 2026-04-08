#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combination.py - 角色卡组合生成器

根据constraints.json中的风险点要求，组合demographics、psychostate、socialcontext
三个schema的元素，生成rawcards.jsonl文件。
"""

import json
import itertools
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Set, Tuple


def load_json(filepath: str) -> Dict:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data: List[Dict], filepath: str):
    """保存为JSONL格式"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_attr_values(schema: Dict, attr_id: str) -> List[str]:
    """从schema中获取指定属性的所有可能值"""
    for attr in schema.get('attributes', []):
        if attr.get('attr_id') == attr_id:
            values = attr.get('values', [])
            return [v.get('value_id') for v in values if v.get('value_id')]
    return []


def get_attr_info(schema: Dict, attr_id: str) -> Optional[Dict]:
    """获取属性的完整信息"""
    for attr in schema.get('attributes', []):
        if attr.get('attr_id') == attr_id:
            return attr
    return None


def is_valid_age_occupation_combination(age: str, occupation: str) -> bool:
    """
    验证年龄-职业组合是否合理
    
    规则：
    - preschool/school_age → 只能是 student 或无职业
    - adolescent → 只能是 student
    - young_adult → 不能是 retired
    - elderly → 不能是 student（极少数例外）
    - retired → 年龄必须 ≥ late_middle_aged（56+）
    """
    # preschool/school_age 只能是 student 或 unemployed(无职业)
    if age in ['preschool', 'school_age']:
        if occupation not in ['student', 'unemployed']:
            return False
    
    # adolescent 只能是 student
    if age == 'adolescent':
        if occupation != 'student':
            return False
    
    # young_adult 不能是 retired
    if age == 'young_adult':
        if occupation == 'retired':
            return False
    
    # elderly 不能是 student
    if age == 'elderly':
        if occupation == 'student':
            return False
    
    # retired 年龄必须 >= late_middle_aged
    if occupation == 'retired':
        valid_ages_for_retired = ['late_middle_aged', 'elderly']
        if age not in valid_ages_for_retired:
            return False
    
    return True


def filter_valid_combinations(combinations: List[Dict]) -> List[Dict]:
    """过滤掉不符合年龄-职业规则的组合"""
    valid_combinations = []
    for combo in combinations:
        age = combo.get('age')
        occupation = combo.get('occupation')
        
        # 如果组合中没有age或occupation，直接通过
        if age is None or occupation is None:
            valid_combinations.append(combo)
            continue
        
        # 验证年龄-职业组合
        if is_valid_age_occupation_combination(age, occupation):
            valid_combinations.append(combo)
    
    return valid_combinations


def extract_schema_elements(constraint_elements: Dict, schemas: Dict) -> Dict[str, Dict[str, List[str]]]:
    """
    从约束条件中提取schema元素及其可能的值
    
    返回格式: {
        'schema_name': {
            'attr_id': [possible_values]
        }
    }
    """
    elements = {}
    
    for schema_name, schema_constraint in constraint_elements.items():
        if schema_name not in schemas:
            continue
            
        schema = schemas[schema_name]
        elements[schema_name] = {}
        
        # 处理required元素
        for req in schema_constraint.get('required', []):
            attr_id = req.get('attr_id')
            if not attr_id:
                continue
            
            # 优先使用high_risk_values，其次是relevant_values/suggested_values，最后是全部值
            if req.get('high_risk_values'):
                values = req['high_risk_values']
            elif req.get('relevant_values'):
                values = req['relevant_values']
            elif req.get('suggested_values'):
                values = req['suggested_values']
            elif req.get('all_values_relevant'):
                values = get_attr_values(schema, attr_id)
            else:
                values = get_attr_values(schema, attr_id)
            
            elements[schema_name][attr_id] = values
        
        # 处理optional元素
        for opt in schema_constraint.get('optional', []):
            attr_id = opt.get('attr_id')
            if not attr_id:
                continue
            
            # optional元素使用全部可能值
            if opt.get('high_risk_values'):
                values = opt['high_risk_values']
            elif opt.get('relevant_values'):
                values = opt['relevant_values']
            else:
                values = get_attr_values(schema, attr_id)
            
            elements[schema_name][attr_id] = values
    
    return elements


def generate_combinations(elements: Dict[str, Dict[str, List[str]]], max_count: int = 4000) -> List[Dict]:
    """
    生成所有属性值的组合
    
    为了控制组合数量，采用分层采样策略
    """
    # 将所有属性展平为 [(attr_key, values), ...]
    all_attrs = []
    for schema_name, attrs in elements.items():
        for attr_id, values in attrs.items():
            if values:  # 确保有值
                all_attrs.append((f"{schema_name}.{attr_id}", values))
    
    if not all_attrs:
        return []
    
    # 计算总组合数
    total_combinations = 1
    for _, values in all_attrs:
        total_combinations *= len(values)
    
    # 如果总数小于max_count，生成所有组合
    if total_combinations <= max_count:
        # 生成笛卡尔积
        keys = [k for k, _ in all_attrs]
        value_lists = [v for _, v in all_attrs]
        
        combinations = []
        for combo in itertools.product(*value_lists):
            combo_dict = {}
            for key, val in zip(keys, combo):
                # 解析 schema_name.attr_id
                parts = key.split('.', 1)
                attr_id = parts[1] if len(parts) > 1 else parts[0]
                combo_dict[attr_id] = val
            combinations.append(combo_dict)
        
        return combinations
    
    # 如果总数太大，进行采样
    combinations = []
    keys = [k for k, _ in all_attrs]
    value_lists = [v for _, v in all_attrs]
    
    # 使用随机采样
    seen = set()
    attempts = 0
    max_attempts = max_count * 10  # 防止无限循环
    
    while len(combinations) < max_count and attempts < max_attempts:
        attempts += 1
        
        # 随机选择每个属性的值
        combo = tuple(random.choice(values) for values in value_lists)
        
        if combo not in seen:
            seen.add(combo)
            combo_dict = {}
            for key, val in zip(keys, combo):
                parts = key.split('.', 1)
                attr_id = parts[1] if len(parts) > 1 else parts[0]
                combo_dict[attr_id] = val
            combinations.append(combo_dict)
    
    return combinations


def process_dimension(
    category_id: str,
    category_name: str,
    dimension_id: str,
    dimension_data: Dict,
    schemas: Dict,
    max_per_dimension: int = 4000
) -> List[Dict]:
    """处理单个风险维度，生成角色卡"""
    
    dimension_name = dimension_data.get('dimension_name', '')
    test_point = dimension_data.get('test_point', '')
    risk_description = dimension_data.get('risk_description', '')
    required_elements = dimension_data.get('required_schema_elements', {})
    
    # 提取schema元素
    elements = extract_schema_elements(required_elements, schemas)
    
    # 生成组合
    combinations = generate_combinations(elements, max_per_dimension)
    
    # 过滤不符合年龄-职业规则的组合
    valid_combinations = filter_valid_combinations(combinations)
    
    # 如果过滤后数量超过限制，进行采样
    if len(valid_combinations) > max_per_dimension:
        valid_combinations = random.sample(valid_combinations, max_per_dimension)
    
    # 构建结果
    results = []
    for combo in valid_combinations:
        card = {
            'category_id': category_id,
            'category_name': category_name,
            'dimension_id': dimension_id,
            'dimension_name': dimension_name,
            'test_point': test_point,
            'risk_description': risk_description,
            'values': combo
        }
        results.append(card)
    
    return results


def main():
    """主函数"""
    # 加载配置
    print("加载schema文件...")
    constraints = load_json('./role-card/schema/constraints.json')
    schemas = {
        'demographics': load_json('./role-card/schema/demographics.json'),
        'psychostate': load_json('./role-card/schema/psychostate.json'),
        'socialcontext': load_json('./role-card/schema/socialcontext.json')
    }
    
    # 统计总维度数
    total_dimensions = 0
    for category_id, category_data in constraints.get('constraints', {}).items():
        total_dimensions += len(category_data.get('dimensions', {}))
    
    print(f"共有 {total_dimensions} 个风险维度")
    
    # 每个维度的最大条目数
    max_per_dimension = 500000 // total_dimensions
    print(f"每个维度最大生成 {max_per_dimension} 条记录")
    
    all_cards = []
    
    # 遍历所有类别和维度
    for category_id, category_data in tqdm(constraints.get('constraints', {}).items(), desc="处理类别"):
        category_name = category_data.get('category_name', '')
        dimensions = category_data.get('dimensions', {})
        
        for dimension_id, dimension_data in tqdm(dimensions.items(), desc=f"  {category_id} 维度", leave=False):
            cards = process_dimension(
                category_id=category_id,
                category_name=category_name,
                dimension_id=dimension_id,
                dimension_data=dimension_data,
                schemas=schemas,
                max_per_dimension=max_per_dimension
            )
            all_cards.extend(cards)
            print(f"    {category_id}-{dimension_id}: 生成 {len(cards)} 条 (过滤后)")
    
    print(f"\n总共生成 {len(all_cards)} 条角色卡")
    
    # 如果超过5万条，进行最终采样
    if len(all_cards) > 50000:
        print(f"总数超过50000，进行采样...")
        all_cards = random.sample(all_cards, 50000)
    
    # 保存结果
    output_path = './role-card/rawcards.jsonl'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_cards, output_path)
    print(f"已保存到 {output_path}")
    
    # 输出统计信息
    print("\n=== 统计信息 ===")
    category_stats = {}
    dimension_stats = {}
    
    for card in all_cards:
        cat_id = card['category_id']
        dim_id = card['dimension_id']
        key = f"{cat_id}-{dim_id}"
        
        category_stats[cat_id] = category_stats.get(cat_id, 0) + 1
        dimension_stats[key] = dimension_stats.get(key, 0) + 1
    
    print("\n按类别统计:")
    for cat_id, count in sorted(category_stats.items()):
        print(f"  {cat_id}: {count} 条")
    
    print("\n按维度统计:")
    for dim_key, count in sorted(dimension_stats.items()):
        print(f"  {dim_key}: {count} 条")


if __name__ == '__main__':
    random.seed(42)
    main()