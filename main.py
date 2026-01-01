"""
create_agent の response_format がエージェント動作に与える影響を検証

検証する仮説:
  ProviderStrategy + validation_result フィールドありのスキーマ
  → LLM が validate_calculation を積極的に呼ぶようになる
"""

import os
from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

from util import (
    tracker, ALL_TOOLS, TEST_CASES,
    SchemaWithValidation, SchemaSimple,
    SYSTEM_PROMPT, SYSTEM_PROMPT_NO_FORMAT_SIMPLE,
)

Mode = Literal["no_format", "tool_strategy", "provider_strategy"]
SchemaType = Literal["with_validation", "simple"]


@dataclass
class TestResult:
    mode: Mode
    schema_type: SchemaType
    validate_calls: int
    success: bool


def get_response_format(mode: Mode, schema_type: SchemaType):
    """モードとスキーマタイプに応じた response_format を返す"""
    if mode == "no_format":
        return None
    schema = SchemaWithValidation if schema_type == "with_validation" else SchemaSimple
    if mode == "tool_strategy":
        return ToolStrategy(schema=schema)
    return ProviderStrategy(schema=schema)


# グローバルLLMインスタンス（再利用で高速化）
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    return _llm


def run_test(mode: Mode, schema_type: SchemaType | None, task: str) -> TestResult:
    """単一テストを実行"""
    tracker.reset()
    
    llm = get_llm()
    # no_format は validation_result を含まない JSON を要求（公平な比較のため）
    prompt = SYSTEM_PROMPT_NO_FORMAT_SIMPLE if mode == "no_format" else SYSTEM_PROMPT
    
    try:
        agent = create_agent(
            llm, ALL_TOOLS,
            response_format=get_response_format(mode, schema_type),
            system_prompt=prompt,
        )
        agent.invoke({"messages": [("user", task)]})
        return TestResult(mode, schema_type, tracker.count("validate_calculation"), True)
    except Exception:
        return TestResult(mode, schema_type, tracker.count("validate_calculation"), False)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY が未設定")
        return

    configs = [
        ("no_format", None),  # response_format なし（スキーマ不使用）
        ("tool_strategy", "with_validation"),
        ("provider_strategy", "with_validation"),
    ]
    
    # 結果格納
    results: dict[str, list[int]] = {
        "no_format": [],
        "tool_strategy": [],
        "provider_strategy": [],
    }
    
    total = len(configs) * len(TEST_CASES)
    print(f"テスト実行中 ({total} 件, {len(TEST_CASES)} テストケース × {len(configs)} 設定)...")
    
    idx = 0
    for mode, schema_type in configs:
        for tc in TEST_CASES:
            idx += 1
            print(f"\r  [{idx}/{total}]", end="", flush=True)
            r = run_test(mode, schema_type, tc.task)
            results[mode].append(r.validate_calls)
    
    print("\n")
    
    # 結果出力
    print("=" * 60)
    print("結果: validate_calculation 呼び出し回数")
    print("=" * 60)
    print(f"{'設定':<35} {'各試行':>12} {'平均':>8}")
    print("-" * 60)
    for key, counts in results.items():
        avg = sum(counts) / len(counts) if counts else 0
        print(f"{key:<35} {str(counts):>12} {avg:>7.1f}")
    
    # 結論
    print("\n" + "=" * 60)
    print("結論")
    print("=" * 60)
    
    n = len(TEST_CASES)
    pv = sum(results.get("provider_strategy", [])) / n
    tv = sum(results.get("tool_strategy", [])) / n
    nf = sum(results.get("no_format", [])) / n
    
    # 分析: 3つのモードを比較
    print("\n📊 分析: ProviderStrategy vs ToolStrategy vs no_format")
    print("-" * 50)
    print(f"   provider_strategy (with_validation): {pv:.1f}回")
    print(f"   tool_strategy (with_validation): {tv:.1f}回")
    print(f"   no_format (スキーマなし): {nf:.1f}回")
    
    if pv > tv:
        print("\n✅ 記事の主張が確認された:")
        print("   → ProviderStrategy は validation_result を埋めようとする")
        print("   → ToolStrategy は Optional フィールドを省略可能")
    elif pv == tv and pv >= 1.0:
        print("\n⚠️  両ストラテジーで同等に validate が呼ばれる")
        print("   → 記事の ToolStrategy の主張は確認できなかった")
    else:
        print("\n❌ 予想外の結果")


if __name__ == "__main__":
    main()
