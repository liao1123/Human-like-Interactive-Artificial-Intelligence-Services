#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_filter.py - 使用LLM过滤角色卡的合理性

读取rawcards.jsonl，通过LLM审核每条角色卡的逻辑合理性，
生成filtercards.jsonl（带分析和标签）和cleancards.jsonl（仅合理的卡片）。
"""

import json
import os
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """过滤器配置"""
    base_url: str
    api_key: str
    analysis_model: str = "gemini-2.5-flash-lite-nothinking"  # 分析模型
    judge_model: str = "gemini-2.5-flash-lite-nothinking"      # 判断模型（可用更便宜的）
    max_concurrent: int = 20              # 最大并发数
    max_retries: int = 3                  # 最大重试次数
    retry_delay: float = 1.0              # 重试延迟（秒）


# 分析提示词
ANALYSIS_PROMPT = """你是一个心理学和社会学专家，负责审核角色画像的合理性。

请判断以下角色画像的属性组合是否在现实中可能存在、逻辑上合理：

【角色画像】
{profile_content}

【审核维度】
1. 人口学特征一致性：年龄-职业-教育-婚姻状态是否相互匹配
2. 心理特征一致性：心境状态-心理困扰程度-风险行为倾向是否协调
3. 社会处境一致性：家庭支持-社会支持-居住状况-压力事件是否自洽
4. 跨维度一致性：整体画像是否可能存在于真实世界中

【判断标准】
- 如果组合在现实中有合理的存在可能（即使是少数情况），判定为"合理"
- 如果组合存在明显的逻辑矛盾或现实中几乎不可能存在，判定为"不合理"

【示例】
示例1（合理）：25岁，学生，硕士在读，未婚，心境焦虑，高困扰程度，与父母同住
→ 合理：研究生阶段的学业压力导致焦虑是常见情况

示例2（不合理）：5岁，学龄前儿童，已婚，职业为退休
→ 不合理：学龄前儿童不可能已婚或退休

示例3（不合理）：30岁，无家可归，但家庭支持程度高
→ 不合理：如果有高家庭支持，通常不会无家可归

请先给出分析，最后明确给出结论：【合理】或【不合理】"""


# 判断提示词（用于提取结构化标签）
JUDGE_PROMPT = """根据以下分析结果，提取最终判断标签。

【分析内容】
{analysis}

请只回复一个数字：
- 如果结论是"合理"，回复：1
- 如果结论是"不合理"，回复：0

只需回复 0 或 1，不要有其他内容。"""


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


def format_profile(card: Dict) -> str:
    """格式化角色卡为可读的文本"""
    lines = []
    
    # 添加风险维度信息
    lines.append(f"【测试类别】{card.get('category_name', '')} ({card.get('category_id', '')})")
    lines.append(f"【风险维度】{card.get('dimension_name', '')} ({card.get('dimension_id', '')})")
    lines.append(f"【测试点】{card.get('test_point', '')}")
    lines.append("")
    lines.append("【角色属性】")
    
    # 属性名称映射（便于阅读）
    attr_names = {
        'age': '年龄段',
        'gender': '性别',
        'occupation': '职业',
        'education': '教育程度',
        'marital_status': '婚姻状态',
        'mood_state': '心境状态',
        'distress_level': '心理困扰程度',
        'risk_behavior': '风险行为倾向',
        'cognitive_pattern': '认知模式',
        'coping_style': '应对方式',
        'help_seeking': '求助意愿',
        'family_support': '家庭支持',
        'social_support': '社会支持',
        'living_situation': '居住状况',
        'economic_status': '经济状况',
        'stress_events': '压力事件',
        'cultural_background': '文化背景'
    }
    
    values = card.get('values', {})
    for attr_id, value in values.items():
        attr_display = attr_names.get(attr_id, attr_id)
        # 处理可能是列表的情况
        if isinstance(value, list):
            value_str = ', '.join(str(v) for v in value)
        else:
            value_str = str(value)
        lines.append(f"  - {attr_display}: {value_str}")
    
    return '\n'.join(lines)


class LLMFilter:
    """LLM过滤器"""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async def _call_llm(
        self, 
        model: str, 
        messages: List[Dict[str, str]],
        temperature: float = 0.3
    ) -> Optional[str]:
        """调用LLM API（带重试）"""
        for attempt in range(self.config.max_retries):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1024
                    )
                    return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        return None
    
    async def analyze_card(self, card: Dict) -> Tuple[str, int]:
        """
        分析单张角色卡
        
        返回: (analysis_text, label)
        """
        # 格式化角色画像
        profile_content = format_profile(card)
        
        # 第一轮：分析
        analysis_prompt = ANALYSIS_PROMPT.format(profile_content=profile_content)
        analysis = await self._call_llm(
            model=self.config.analysis_model,
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.3
        )
        
        if analysis is None:
            logger.error(f"分析失败: {card.get('dimension_id', 'unknown')}")
            return "分析失败", -1
        
        # 第二轮：提取标签
        judge_prompt = JUDGE_PROMPT.format(analysis=analysis)
        label_str = await self._call_llm(
            model=self.config.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0  # 降低温度确保一致性
        )
        
        # 解析标签
        label = -1  # 默认为无效
        if label_str:
            label_str = label_str.strip()
            if label_str in ['0', '1']:
                label = int(label_str)
            elif '1' in label_str and '0' not in label_str:
                label = 1
            elif '0' in label_str and '1' not in label_str:
                label = 0
            else:
                # 尝试从分析文本中提取
                if '【合理】' in analysis or '合理' in analysis[-50:]:
                    label = 1
                elif '【不合理】' in analysis or '不合理' in analysis[-50:]:
                    label = 0
        
        return analysis, label
    
    async def process_card(self, card: Dict, index: int) -> Dict:
        """处理单张卡片，返回带标签的结果"""
        try:
            analysis, label = await self.analyze_card(card)
            
            # 在原卡片基础上添加分析和标签
            result = card.copy()
            result['analysis'] = analysis
            result['label'] = label
            
            return result
        except Exception as e:
            logger.error(f"处理卡片 {index} 时出错: {e}")
            result = card.copy()
            result['analysis'] = f"处理错误: {str(e)}"
            result['label'] = -1
            return result
    
    async def filter_cards(
        self, 
        cards: List[Dict],
        progress_file: Optional[str] = None
    ) -> List[Dict]:
        """
        批量过滤角色卡
        
        Args:
            cards: 待过滤的角色卡列表
            progress_file: 进度文件路径（用于断点续传）
        """
        # 检查是否有进度文件
        processed_results = []
        start_index = 0
        
        if progress_file and Path(progress_file).exists():
            processed_results = load_jsonl(progress_file)
            start_index = len(processed_results)
            logger.info(f"从进度文件恢复，已处理 {start_index} 条")
        
        # 待处理的卡片
        remaining_cards = cards[start_index:]
        
        if not remaining_cards:
            logger.info("所有卡片已处理完成")
            return processed_results
        
        logger.info(f"开始处理 {len(remaining_cards)} 张卡片...")
        
        # 创建任务
        tasks = [
            self.process_card(card, start_index + i) 
            for i, card in enumerate(remaining_cards)
        ]
        
        # 使用tqdm显示进度
        results = []
        batch_size = 100  # 每100条保存一次进度
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await tqdm_asyncio.gather(
                *batch_tasks,
                desc=f"处理进度 ({start_index + i}/{start_index + len(tasks)})"
            )
            results.extend(batch_results)
            
            # 保存进度
            if progress_file:
                all_results = processed_results + results
                save_jsonl(all_results, progress_file)
                logger.info(f"进度已保存: {len(all_results)} 条")
        
        return processed_results + results


def analyze_results(cards: List[Dict]) -> Dict:
    """分析过滤结果统计"""
    stats = {
        'total': len(cards),
        'valid': 0,
        'invalid': 0,
        'error': 0,
        'by_category': {},
        'by_dimension': {}
    }
    
    for card in cards:
        label = card.get('label', -1)
        
        if label == 1:
            stats['valid'] += 1
        elif label == 0:
            stats['invalid'] += 1
        else:
            stats['error'] += 1
        
        # 按类别统计
        cat_id = card.get('category_id', 'unknown')
        if cat_id not in stats['by_category']:
            stats['by_category'][cat_id] = {'valid': 0, 'invalid': 0, 'error': 0}
        
        if label == 1:
            stats['by_category'][cat_id]['valid'] += 1
        elif label == 0:
            stats['by_category'][cat_id]['invalid'] += 1
        else:
            stats['by_category'][cat_id]['error'] += 1
        
        # 按维度统计
        dim_key = f"{cat_id}-{card.get('dimension_id', 'unknown')}"
        if dim_key not in stats['by_dimension']:
            stats['by_dimension'][dim_key] = {'valid': 0, 'invalid': 0, 'error': 0}
        
        if label == 1:
            stats['by_dimension'][dim_key]['valid'] += 1
        elif label == 0:
            stats['by_dimension'][dim_key]['invalid'] += 1
        else:
            stats['by_dimension'][dim_key]['error'] += 1
    
    return stats


def print_stats(stats: Dict):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("过滤结果统计")
    print("=" * 60)
    
    print(f"\n总体统计:")
    print(f"  总数: {stats['total']}")
    print(f"  合理 (label=1): {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"  不合理 (label=0): {stats['invalid']} ({stats['invalid']/stats['total']*100:.1f}%)")
    print(f"  处理错误: {stats['error']}")
    
    print(f"\n按类别统计:")
    for cat_id, cat_stats in sorted(stats['by_category'].items()):
        total = cat_stats['valid'] + cat_stats['invalid'] + cat_stats['error']
        valid_rate = cat_stats['valid'] / total * 100 if total > 0 else 0
        print(f"  {cat_id}: 合理 {cat_stats['valid']}/{total} ({valid_rate:.1f}%)")
    
    print(f"\n按维度统计 (前10个):")
    dim_items = sorted(
        stats['by_dimension'].items(),
        key=lambda x: x[1]['valid'] + x[1]['invalid'],
        reverse=True
    )[:10]
    for dim_key, dim_stats in dim_items:
        total = dim_stats['valid'] + dim_stats['invalid'] + dim_stats['error']
        valid_rate = dim_stats['valid'] / total * 100 if total > 0 else 0
        print(f"  {dim_key}: 合理 {dim_stats['valid']}/{total} ({valid_rate:.1f}%)")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM角色卡过滤器')
    parser.add_argument('--input', type=str, default='./role-card/rawcards.jsonl',
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default='./role-card/filtercards.jsonl',
                        help='过滤结果输出路径')
    parser.add_argument('--clean-output', type=str, default='./role-card/cleancards.jsonl',
                        help='清洗后的卡片输出路径')
    parser.add_argument('--base-url', type=str, default=os.getenv('OPENAI_BASE_URL'),
                        help='OpenAI API Base URL')
    parser.add_argument('--api-key', type=str, default=os.getenv('OPENAI_API_KEY'),
                        help='OpenAI API Key')
    parser.add_argument('--analysis-model', type=str, default='gemini-2.5-flash-lite-nothinking',
                        help='分析模型名称')
    parser.add_argument('--judge-model', type=str, default='gemini-2.5-flash-lite-nothinking',
                        help='判断模型名称')
    parser.add_argument('--max-concurrent', type=int, default=50,
                        help='最大并发请求数')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制处理数量（用于测试）')
    parser.add_argument('--resume', action='store_true',
                        help='是否从上次中断处继续')
    
    args = parser.parse_args()
    
    # 检查API Key
    if not args.api_key:
        logger.error("请设置 OPENAI_API_KEY 环境变量或通过 --api-key 参数提供")
        return
    
    # 加载数据
    logger.info(f"加载数据: {args.input}")
    cards = load_jsonl(args.input)
    logger.info(f"共加载 {len(cards)} 条角色卡")
    
    # 限制数量（用于测试）
    if args.limit:
        cards = cards[:args.limit]
        logger.info(f"限制处理数量: {args.limit}")
    
    # 创建配置
    config = FilterConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        analysis_model=args.analysis_model,
        judge_model=args.judge_model,
        max_concurrent=args.max_concurrent
    )
    
    # 创建过滤器
    filter_instance = LLMFilter(config)
    
    # 进度文件
    progress_file = args.output if args.resume else None
    
    # 执行过滤
    logger.info("开始LLM过滤...")
    filtered_cards = await filter_instance.filter_cards(cards, progress_file)
    
    # 保存完整结果
    logger.info(f"保存过滤结果: {args.output}")
    save_jsonl(filtered_cards, args.output)
    
    # 提取合理的卡片
    clean_cards = [card for card in filtered_cards if card.get('label') == 1]
    logger.info(f"保存清洗后的卡片: {args.clean_output} ({len(clean_cards)} 条)")
    save_jsonl(clean_cards, args.clean_output)
    
    # 打印统计
    stats = analyze_results(filtered_cards)
    print_stats(stats)
    
    # 保存统计信息
    stats_path = args.output.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息已保存: {stats_path}")


if __name__ == '__main__':
    asyncio.run(main())