"""
ユーティリティモジュール - create_agent response_format 検証用
"""

from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ============================================================================
# ツール呼び出しトラッカー
# ============================================================================

@dataclass
class ToolCallTracker:
    """ツール呼び出しをカウントするシンプルなトラッカー"""
    _counts: dict[str, int] = field(default_factory=dict)
    
    def reset(self):
        self._counts = {}
    
    def record(self, tool_name: str):
        self._counts[tool_name] = self._counts.get(tool_name, 0) + 1
    
    def count(self, tool_name: str) -> int:
        return self._counts.get(tool_name, 0)
    
    def summary(self) -> dict[str, int]:
        return dict(self._counts)


tracker = ToolCallTracker()


# ============================================================================
# ツール定義
# ============================================================================

@tool
def add(a: float, b: float) -> float:
    """2つの数値を加算します。"""
    tracker.record("add")
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """2つの数値を減算します (a - b)。"""
    tracker.record("subtract")
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """2つの数値を乗算します。"""
    tracker.record("multiply")
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """2つの数値を除算します (a / b)。"""
    if b == 0:
        raise ValueError("ゼロ除算不可")
    tracker.record("divide")
    return a / b


@tool
def validate_calculation(expected: float, actual: float, operation: str) -> dict:
    """計算結果を検証します。"""
    tracker.record("validate_calculation")
    is_correct = abs(expected - actual) < 1e-6
    return {"operation": operation, "is_correct": is_correct}


ALL_TOOLS = [add, subtract, multiply, divide, validate_calculation]


# ============================================================================
# スキーマ定義
# ============================================================================

class SchemaWithValidation(BaseModel):
    """validation_result フィールドを含むスキーマ"""
    final_answer: float = Field(description="最終的な計算結果")
    validation_result: dict | None = Field(
        default=None, description="validate_calculation の結果"
    )
    reasoning: str = Field(description="計算プロセスの説明")


class SchemaSimple(BaseModel):
    """validation_result なしのシンプルなスキーマ"""
    final_answer: float = Field(description="最終的な計算結果")
    reasoning: str = Field(description="計算プロセスの説明")


# ============================================================================
# テストケース
# ============================================================================

@dataclass
class TestCase:
    name: str
    task: str


# 検証を明示的に指示しないタスク（スキーマの影響を見るため）
TEST_CASES = [
    TestCase("calc_01", "(17 × 23) + (89 ÷ 4) - 156 を計算してください。"),           # 257.25
    TestCase("calc_02", "(1024 ÷ 16) × 7 - (33 + 45) を計算してください。"),          # 370
    TestCase("calc_03", "(999 - 123) × 2 ÷ 4 + 67 を計算してください。"),             # 505
    TestCase("calc_04", "(48 + 72) × 5 - (200 ÷ 8) を計算してください。"),            # 575
    TestCase("calc_05", "(144 ÷ 12) + (35 × 6) - 89 を計算してください。"),           # 133
    TestCase("calc_06", "(500 - 123) × 3 ÷ 9 + 44 を計算してください。"),             # 169.67
    TestCase("calc_07", "(81 ÷ 9) × (14 + 6) - 55 を計算してください。"),             # 125
    TestCase("calc_08", "(256 + 128) ÷ 4 × 3 - 100 を計算してください。"),            # 188
    TestCase("calc_09", "(77 × 11) - (324 ÷ 18) + 29 を計算してください。"),          # 858
    TestCase("calc_10", "(1000 - 450) ÷ 5 + (32 × 4) を計算してください。"),          # 238
]


# ============================================================================
# システムプロンプト
# ============================================================================

SYSTEM_PROMPT = """あなたは計算を実行するアシスタントです。

【ルール】
1. 計算は必ず add, subtract, multiply, divide ツールを使用
2. 暗算禁止。必ずツールを呼ぶこと
3. ステップごとにツールを呼び出すこと
"""

# no_format: validation_result なしのJSON形式を要求（公平な比較のため）
SYSTEM_PROMPT_NO_FORMAT_SIMPLE = """あなたは計算を実行するアシスタントです。

【ルール】
1. 計算は必ず add, subtract, multiply, divide ツールを使用
2. 暗算禁止。必ずツールを呼ぶこと
3. ステップごとにツールを呼び出すこと

【出力形式】
最終回答はJSON形式で:
{"final_answer": <数値>, "reasoning": "<説明>"}
"""
