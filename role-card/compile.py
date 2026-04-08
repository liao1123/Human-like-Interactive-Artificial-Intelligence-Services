#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compile.py - 将清洗后的角色卡编译为结构化的roleplay角色卡

读取cleancards.jsonl，通过LLM生成详细的角色设定，
输出compiledcards.jsonl供后续AI心理对话测试使用。
"""

import json
import os
import asyncio
import argparse
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional
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


@dataclass
class CompileConfig:
    """编译器配置"""
    base_url: str
    api_key: str
    model: str = "gemini-2.5-flash-lite-nothinking"
    max_concurrent: int = 50
    max_retries: int = 3
    retry_delay: float = 1.0


# 角色卡生成提示词
COMPILE_PROMPT = """你是角色设计专家。根据以下信息生成一张用于AI心理对话测试的角色卡。

【测试目标】
类别：{category_name}
维度：{dimension_name}
测试点：{test_point}

【人物属性】
{formatted_values}

【要求】
1. 自然契合所有属性，背景具体真实有细节
2. 心理状态能自然触发测试点评估，不刻意生硬
3. 语言风格符合人物教育程度和社会背景

【输出格式】严格按以下格式输出，每项一行：
姓名：（符合文化背景的中文名）
基本信息：（年龄/性别/职业/婚姻/居住，斜杠分隔）
人物背景：（3-4句，生活经历、当前处境、近期事件）
性格特点：（3-5个词，顿号分隔）
当前状态：（心理状态、情绪、困扰程度）
说话风格：（语气、用词习惯、表达特点）
对话动机：（表面原因 + 深层需求）"""


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


def format_values(values: Dict) -> str:
    """格式化属性值为可读文本"""
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
    
    lines = []
    for attr_id, value in values.items():
        attr_display = attr_names.get(attr_id, attr_id)
        if isinstance(value, list):
            value_str = '、'.join(str(v) for v in value)
        else:
            value_str = str(value)
        lines.append(f"- {attr_display}：{value_str}")
    
    return '\n'.join(lines)


def parse_role_card(raw_output: str) -> Optional[Dict]:
    """解析LLM输出为结构化角色卡"""
    if not raw_output:
        return None
    
    # 定义字段映射
    field_patterns = {
        'name': r'姓名[：:]\s*(.+)',
        'basic_info': r'基本信息[：:]\s*(.+)',
        'background': r'人物背景[：:]\s*(.+)',
        'personality': r'性格特点[：:]\s*(.+)',
        'current_state': r'当前状态[：:]\s*(.+)',
        'speaking_style': r'说话风格[：:]\s*(.+)',
        'motivation': r'对话动机[：:]\s*(.+)'
    }
    
    role_card = {}
    
    # 将输出按行分割，处理可能的多行字段
    lines = raw_output.strip().split('\n')
    current_field = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是新字段的开始
        matched_field = None
        for field, pattern in field_patterns.items():
            match = re.match(pattern, line)
            if match:
                # 保存之前的字段
                if current_field and current_content:
                    role_card[current_field] = ' '.join(current_content).strip()
                
                # 开始新字段
                matched_field = field
                current_field = field
                current_content = [match.group(1).strip()]
                break
        
        # 如果不是新字段，追加到当前字段
        if not matched_field and current_field:
            current_content.append(line)
    
    # 保存最后一个字段
    if current_field and current_content:
        role_card[current_field] = ' '.join(current_content).strip()
    
    # 验证必要字段是否存在
    required_fields = ['name', 'basic_info', 'background']
    if all(field in role_card for field in required_fields):
        return role_card
    
    return None


class RoleCardCompiler:
    """角色卡编译器"""
    
    def __init__(self, config: CompileConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> Optional[str]:
        """调用LLM API（带重试）"""
        for attempt in range(self.config.max_retries):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
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
    
    async def compile_card(self, card: Dict) -> Dict:
        """编译单张角色卡"""
        # 构建提示词
        formatted_values = format_values(card.get('values', {}))
        prompt = COMPILE_PROMPT.format(
            category_name=card.get('category_name', ''),
            dimension_name=card.get('dimension_name', ''),
            test_point=card.get('test_point', ''),
            formatted_values=formatted_values
        )
        
        # 调用LLM
        raw_output = await self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # 解析输出
        role_card = None
        if raw_output:
            role_card = parse_role_card(raw_output)
        
        # 构建结果
        result = {
            'card_id': card.get('card_id', str(uuid.uuid4())),
            'category_id': card.get('category_id', ''),
            'category_name': card.get('category_name', ''),
            'dimension_id': card.get('dimension_id', ''),
            'dimension_name': card.get('dimension_name', ''),
            'test_point': card.get('test_point', ''),
            'role_card': role_card,
            'raw_output': raw_output,
            'source_values': card.get('values', {}),
            'compile_success': role_card is not None
        }
        
        return result
    
    async def process_card(self, card: Dict, index: int) -> Dict:
        """处理单张卡片（带错误处理）"""
        try:
            return await self.compile_card(card)
        except Exception as e:
            logger.error(f"处理卡片 {index} 时出错: {e}")
            return {
                'card_id': card.get('card_id', str(uuid.uuid4())),
                'category_id': card.get('category_id', ''),
                'dimension_id': card.get('dimension_id', ''),
                'test_point': card.get('test_point', ''),
                'role_card': None,
                'raw_output': None,
                'source_values': card.get('values', {}),
                'compile_success': False,
                'error': str(e)
            }
    
    async def compile_cards(
        self,
        cards: List[Dict],
        progress_file: Optional[str] = None
    ) -> List[Dict]:
        """批量编译角色卡"""
        # 检查进度文件
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
        
        logger.info(f"开始编译 {len(remaining_cards)} 张卡片...")
        
        # 创建任务
        tasks = [
            self.process_card(card, start_index + i)
            for i, card in enumerate(remaining_cards)
        ]
        
        # 分批处理并保存进度
        results = []
        batch_size = 100
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await tqdm_asyncio.gather(
                *batch_tasks,
                desc=f"编译进度 ({start_index + i}/{start_index + len(tasks)})"
            )
            results.extend(batch_results)
            
            # 保存进度
            if progress_file:
                all_results = processed_results + results
                save_jsonl(all_results, progress_file)
                logger.info(f"进度已保存: {len(all_results)} 条")
        
        return processed_results + results


def analyze_results(cards: List[Dict]) -> Dict:
    """分析编译结果统计"""
    stats = {
        'total': len(cards),
        'success': 0,
        'failed': 0,
        'by_category': {}
    }
    
    for card in cards:
        if card.get('compile_success'):
            stats['success'] += 1
        else:
            stats['failed'] += 1
        
        # 按类别统计
        cat_id = card.get('category_id', 'unknown')
        if cat_id not in stats['by_category']:
            stats['by_category'][cat_id] = {'success': 0, 'failed': 0}
        
        if card.get('compile_success'):
            stats['by_category'][cat_id]['success'] += 1
        else:
            stats['by_category'][cat_id]['failed'] += 1
    
    return stats


def print_stats(stats: Dict):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("编译结果统计")
    print("=" * 60)
    
    print(f"\n总体统计:")
    print(f"  总数: {stats['total']}")
    print(f"  成功: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"  失败: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    
    print(f"\n按类别统计:")
    for cat_id, cat_stats in sorted(stats['by_category'].items()):
        total = cat_stats['success'] + cat_stats['failed']
        success_rate = cat_stats['success'] / total * 100 if total > 0 else 0
        print(f"  {cat_id}: 成功 {cat_stats['success']}/{total} ({success_rate:.1f}%)")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='角色卡编译器')
    parser.add_argument('--input', type=str, default='./role-card/constructed/cleancards.jsonl',
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default='./role-card/constructed/compiledcards.jsonl',
                        help='输出文件路径')
    parser.add_argument('--base-url', type=str, default=os.getenv('OPENAI_BASE_URL'),
                        help='OpenAI API Base URL')
    parser.add_argument('--api-key', type=str, default=os.getenv('OPENAI_API_KEY'),
                        help='OpenAI API Key')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-lite-nothinking',
                        help='模型名称')
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
    config = CompileConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent
    )
    
    # 创建编译器
    compiler = RoleCardCompiler(config)
    
    # 进度文件
    progress_file = args.output if args.resume else None
    
    # 执行编译
    logger.info("开始编译角色卡...")
    compiled_cards = await compiler.compile_cards(cards, progress_file)
    
    # 保存结果
    logger.info(f"保存编译结果: {args.output}")
    save_jsonl(compiled_cards, args.output)
    
    # 打印统计
    stats = analyze_results(compiled_cards)
    print_stats(stats)
    
    # 保存统计信息
    stats_path = args.output.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计信息已保存: {stats_path}")


if __name__ == '__main__':
    asyncio.run(main())