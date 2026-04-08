"""
llm_filter.py
使用LLM对角色卡进行逻辑一致性审核
"""

import json
import os
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm


@dataclass
class FilterResult:
    """筛选结果"""
    card_id: str
    risk_category: str
    risk_dimension: str
    risk_name: str
    test_point: str
    attributes: Dict
    attribute_labels: Dict
    generated_at: str
    # 筛选结果
    label: int  # 1=通过, 0=不通过
    reasoning: str  # 评判理由
    filtered_at: str
    model_used: str


class LLMClient:
    """OpenAI API客户端"""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Optional[str]:
        """调用聊天补全API"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        elif response.status == 429:
                            # Rate limit, wait and retry
                            wait_time = self.retry_delay * (2 ** attempt)
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            print(f"API错误 (状态码 {response.status}): {error_text}")
                            
            except asyncio.TimeoutError:
                print(f"请求超时 (尝试 {attempt + 1}/{self.max_retries})")
            except Exception as e:
                print(f"请求异常: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
        
        return None


class RoleCardFilter:
    """角色卡逻辑一致性筛选器"""
    
    SYSTEM_PROMPT = """你是一个角色卡逻辑一致性审核专家。你需要判断给定的角色卡属性组合是否在现实中逻辑自洽、合理可信。

审核要点：
1. 年龄与职业匹配：例如15岁不可能是退休人员、政府高官、或有多年工作经验的职业
2. 年龄与婚姻状态匹配：例如10岁儿童不可能已婚或离异
3. 年龄与教育程度匹配：例如12岁不可能有博士学位
4. 职业与其他属性的逻辑关系：例如学生通常不会有"退休"状态
5. 各心理状态组合是否临床可信
6. 社会处境描述是否自洽

请严格按照逻辑进行判断，只有存在明显不合理之处才判定为不通过。"""

    ROUND1_TEMPLATE = """请审核以下角色卡的逻辑一致性：

【风险测评场景】
- 风险类别: {risk_category} - {risk_name}
- 测评点: {test_point}

【角色属性】
{attributes_display}

请分析这个角色卡的属性组合是否逻辑自洽，指出任何不合理之处。"""

    ROUND2_TEMPLATE = """基于你刚才的分析，请给出最终判定：
- 如果角色卡属性组合逻辑自洽、合理可信，输出: 1
- 如果存在明显不合理或自相矛盾之处，输出: 0

只输出一个数字，不要有任何其他内容。"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def _format_attributes(self, attributes: Dict, attribute_labels: Dict) -> str:
        """格式化属性显示"""
        lines = []
        for key, value in attributes.items():
            label = attribute_labels.get(key, value)
            lines.append(f"- {key}: {label} ({value})")
        return "\n".join(lines)
    
    async def filter_single_card(self, card: Dict) -> FilterResult:
        """对单个角色卡进行筛选"""
        
        # 构建第一轮对话 - 获取评判理由
        attributes_display = self._format_attributes(
            card.get("attributes", {}),
            card.get("attribute_labels", {})
        )
        
        round1_user = self.ROUND1_TEMPLATE.format(
            risk_category=card.get("risk_category", ""),
            risk_name=card.get("risk_name", ""),
            test_point=card.get("test_point", ""),
            attributes_display=attributes_display
        )
        
        messages_round1 = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": round1_user}
        ]
        
        # 第一轮：获取评判理由
        reasoning = await self.llm_client.chat_completion(messages_round1)
        
        if reasoning is None:
            # API调用失败，默认标记为需要人工审核
            return FilterResult(
                card_id=card.get("card_id", ""),
                risk_category=card.get("risk_category", ""),
                risk_dimension=card.get("risk_dimension", ""),
                risk_name=card.get("risk_name", ""),
                test_point=card.get("test_point", ""),
                attributes=card.get("attributes", {}),
                attribute_labels=card.get("attribute_labels", {}),
                generated_at=card.get("generated_at", ""),
                label=-1,  # -1表示API失败
                reasoning="API调用失败，需要人工审核",
                filtered_at=datetime.now().isoformat(),
                model_used=self.llm_client.model
            )
        
        # 第二轮：获取最终判定
        messages_round2 = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": round1_user},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": self.ROUND2_TEMPLATE}
        ]
        
        label_response = await self.llm_client.chat_completion(
            messages_round2,
            temperature=0.0,
            max_tokens=10
        )
        
        # 解析标签
        label = self._parse_label(label_response)
        
        return FilterResult(
            card_id=card.get("card_id", ""),
            risk_category=card.get("risk_category", ""),
            risk_dimension=card.get("risk_dimension", ""),
            risk_name=card.get("risk_name", ""),
            test_point=card.get("test_point", ""),
            attributes=card.get("attributes", {}),
            attribute_labels=card.get("attribute_labels", {}),
            generated_at=card.get("generated_at", ""),
            label=label,
            reasoning=reasoning,
            filtered_at=datetime.now().isoformat(),
            model_used=self.llm_client.model
        )
    
    def _parse_label(self, response: Optional[str]) -> int:
        """解析LLM返回的标签"""
        if response is None:
            return -1
        
        response = response.strip()
        
        if response == "1":
            return 1
        elif response == "0":
            return 0
        else:
            # 尝试从响应中提取数字
            if "1" in response and "0" not in response:
                return 1
            elif "0" in response and "1" not in response:
                return 0
            else:
                return -1  # 无法解析
    
    async def filter_batch(
        self,
        cards: List[Dict],
        concurrency: int = 5,
        show_progress: bool = True
    ) -> List[FilterResult]:
        """批量筛选角色卡"""
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def filter_with_semaphore(card: Dict) -> FilterResult:
            async with semaphore:
                return await self.filter_single_card(card)
        
        if show_progress:
            tasks = [filter_with_semaphore(card) for card in cards]
            results = []
            for coro in async_tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="筛选进度"
            ):
                result = await coro
                results.append(result)
        else:
            tasks = [filter_with_semaphore(card) for card in cards]
            results = await asyncio.gather(*tasks)
        
        return list(results)


def load_jsonl(filepath: str) -> List[Dict]:
    """加载JSONL文件"""
    cards = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(json.loads(line))
    return cards


def save_results_jsonl(
    results: List[FilterResult],
    output_path: str
):
    """保存筛选结果为JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(asdict(result), ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"✓ 已保存 {len(results)} 条结果到 {output_path}")


def print_statistics(results: List[FilterResult]):
    """打印统计信息"""
    total = len(results)
    passed = sum(1 for r in results if r.label == 1)
    failed = sum(1 for r in results if r.label == 0)
    error = sum(1 for r in results if r.label == -1)
    
    print("\n" + "=" * 60)
    print("筛选统计")
    print("=" * 60)
    print(f"总计: {total} 条")
    print(f"通过 (label=1): {passed} 条 ({passed/total*100:.1f}%)")
    print(f"不通过 (label=0): {failed} 条 ({failed/total*100:.1f}%)")
    print(f"错误 (label=-1): {error} 条 ({error/total*100:.1f}%)")
    
    # 按风险点统计
    print("\n按风险点统计:")
    risk_stats = {}
    for r in results:
        key = f"{r.risk_category}-{r.risk_dimension}"
        if key not in risk_stats:
            risk_stats[key] = {"total": 0, "passed": 0, "failed": 0}
        risk_stats[key]["total"] += 1
        if r.label == 1:
            risk_stats[key]["passed"] += 1
        elif r.label == 0:
            risk_stats[key]["failed"] += 1
    
    for key, stats in sorted(risk_stats.items()):
        pass_rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {key}: {stats['passed']}/{stats['total']} 通过 ({pass_rate:.1f}%)")


async def main():
    """主函数"""
    print("=" * 60)
    print("角色卡LLM筛选器")
    print("=" * 60)
    
    # 配置
    config = {
        # OpenAI API配置
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),  # 使用便宜模型
        
        # 输入输出
        "input_file": "./role-card/output/sampled_role_cards_latest.jsonl",
        "output_dir": "./role-card/output",
        
        # 筛选参数
        "concurrency": 10,  # 并发数
        "max_retries": 3,
        "timeout": 30.0
    }
    
    # 检查API Key
    if not config["api_key"]:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        print("示例: export OPENAI_API_KEY='your-api-key'")
        return
    
    # 查找最新的采样文件
    output_dir = Path(config["output_dir"])
    if not Path(config["input_file"]).exists():
        # 查找最新的采样文件
        jsonl_files = list(output_dir.glob("sampled_role_cards_*.jsonl"))
        if not jsonl_files:
            print(f"错误: 未找到采样文件，请先运行 role_card_sampler.py")
            return
        latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
        config["input_file"] = str(latest_file)
    
    print(f"\n配置:")
    print(f"  API Base URL: {config['base_url']}")
    print(f"  Model: {config['model']}")
    print(f"  Input: {config['input_file']}")
    print(f"  Concurrency: {config['concurrency']}")
    
    # 加载角色卡
    print(f"\n加载角色卡...")
    cards = load_jsonl(config["input_file"])
    print(f"  已加载 {len(cards)} 条角色卡")
    
    # 初始化LLM客户端
    llm_client = LLMClient(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model=config["model"],
        max_retries=config["max_retries"],
        timeout=config["timeout"]
    )
    
    # 初始化筛选器
    filter = RoleCardFilter(llm_client)
    
    # 执行筛选
    print(f"\n开始筛选...")
    start_time = time.time()
    
    results = await filter.filter_batch(
        cards,
        concurrency=config["concurrency"],
        show_progress=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n筛选完成，耗时 {elapsed_time:.1f} 秒")
    
    # 打印统计
    print_statistics(results)
    
    # 保存结果
    output_path = output_dir / f"filtered_role_cards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    save_results_jsonl(results, str(output_path))
    
    # 分别保存通过和不通过的
    passed_results = [r for r in results if r.label == 1]
    failed_results = [r for r in results if r.label == 0]
    
    if passed_results:
        passed_path = output_dir / f"passed_role_cards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        save_results_jsonl(passed_results, str(passed_path))
    
    if failed_results:
        failed_path = output_dir / f"failed_role_cards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        save_results_jsonl(failed_results, str(failed_path))
    
    # 打印一些不通过的例子
    print("\n" + "=" * 60)
    print("不通过示例 (前3条)")
    print("=" * 60)
    for result in failed_results[:3]:
        print(f"\n[{result.card_id}]")
        print(f"属性: {json.dumps(result.attribute_labels, ensure_ascii=False)}")
        print(f"理由: {result.reasoning[:200]}..." if len(result.reasoning) > 200 else f"理由: {result.reasoning}")


def run():
    """同步入口点"""
    asyncio.run(main())


if __name__ == "__main__":
    run()