# create_agent の response_format 動作検証

LangChain v1 の `create_agent` において、`response_format` 引数がエージェントの動作に与える影響を検証するプログラムです。

## 📄 検証した主張

> **「response_format を指定してエビデンスフィールド（validation_result）を含むスキーマを設定すると、LLM が validate 関連のツールを積極的に呼び出すようになる」**

出典: [create_agentのレスポンスフォーマットを使う場合の影響.md](./create_agentのレスポンスフォーマットを使う場合の影響.md)

## 🧪 検証方法

1. **検証を指示しないタスク**で4つの方式を比較
   - `no_format`: response_format なし (ReAct パターン)
   - `tool_strategy`: ToolStrategy (スキーマをツールとして登録)
   - `provider_strategy`: ProviderStrategy (OpenAI native JSON mode) ★明示的に指定
   - `auto_strategy`: AutoStrategy (自動選択)

2. **validation_result フィールドの有無**で比較
   - `CalculationResult`: validation_result フィールドあり
   - `SimpleCalculationResult`: validation_result フィールドなし

3. **validate_calculation の呼び出し回数**を記録（各テスト3回実行）

## 📊 検証結果

### テストケース1: 検証指示なし + validation_result フィールドあり

```
タスク: "15 × 3 を計算してください。"

no_format         : [0, 0, 0] 平均=0.0回
tool_strategy     : [0, 0, 0] 平均=0.0回
provider_strategy : [1, 1, 1] 平均=1.0回 ✅
auto_strategy     : [1, 1, 1] 平均=1.0回 ✅
```

### テストケース3: 検証指示なし + validation_result フィールドなし

```
タスク: "15 × 3 を計算してください。"
スキーマ: SimpleCalculationResult

tool_strategy     : [0, 0, 0] 平均=0.0回
provider_strategy : [0, 0, 0] 平均=0.0回
auto_strategy     : [0, 0, 0] 平均=0.0回
```

## 🎯 結論

### ✅ 記事の主張は **ProviderStrategy において完全に確認できた**

**重要な発見**: ProviderStrategy を**明示的に指定**した場合と AutoStrategy で自動選択された場合の両方で、同じ挙動が確認されました。

1. **response_format の影響**
   - `no_format`: validate 呼び出し 0回
   - `tool_strategy`: validate 呼び出し 0回
   - `provider_strategy`: validate 呼び出し 1回 ✅
   - `auto_strategy`: validate 呼び出し 1回 ✅
   - → **ProviderStrategy は validation_result フィールドを埋めるために validate ツールを呼ぶ**

2. **validation_result フィールドの影響**
   - `provider_strategy` (validation_result あり): 1回
   - `provider_strategy` (validation_result なし): 0回
   - → **スキーマに validation_result フィールドがあると、LLM はそれを埋めるために validate ツールを呼ぶ**

3. **ToolStrategy では同様の傾向は見られなかった**
   - これは ToolStrategy と ProviderStrategy の内部実装の違いによるもの

## 🔍 技術的な洞察

### AutoStrategy の挙動

**gpt-4o-mini では AutoStrategy は ProviderStrategy を選択**
- `_supports_provider_strategy()` が `True` を返すため
- FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT に "gpt-4o" が含まれる
- 実際に検証した結果、すべての実行で ProviderStrategy が選択された

### ProviderStrategy vs ToolStrategy の違い

- **ToolStrategy**: スキーマを「ツール」として登録し、`tool_choice="required"` を強制
  - LLM は明示的にツール呼び出しの形式でスキーマを呼ぶ
  - Optional フィールド（`validation_result: dict | None`）は必ずしも埋めなくて良い
  - 検証では validate ツールを呼ばなかった（0回）

- **ProviderStrategy**: OpenAI の native JSON mode (structured outputs) を使用
  - LLM はスキーマに厳密に準拠した JSON を生成しようとする
  - Optional でもフィールドが定義されていると、値を埋めようとする傾向が強い
  - そのため `validation_result` を埋めるために `validate_calculation` ツールを呼ぶ（1回）
  - これが記事で述べられていた「validate ツールを積極的に呼ぶ」挙動の正体

## 🚀 実行方法

```bash
export OPENAI_API_KEY="your-api-key"
uv run python main.py
```

## 📦 依存関係

- Python 3.13+
- langchain >= 1.2.0
- langchain-openai >= 1.1.6
- openai >= 2.14.0

## 📝 ファイル構成

- `main.py`: 検証プログラム本体
- `create_agentのレスポンスフォーマットを使う場合の影響.md`: 検証対象の記事
- `README.md`: このファイル
