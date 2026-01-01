# Create Agent Investigations

LangChain v1 の `create_agent` における `response_format` 引数の動作を検証し、構造化出力がエージェントの振る舞いに与える影響を調査するプロジェクトです。

## 🎯 プロジェクトの概要

このプロジェクトでは、以下の主張を検証します：

> **「response_format を指定してエビデンスフィールド（validation_result）を含むスキーマを設定すると、LLM が validate 関連のツールを積極的に呼び出すようになる」**

## 🔬 検証結果

詳細な検証方法、結果、および技術的洞察については [investigation.md](./investigation.md) をご覧ください。

## 🚀 クイックスタート

### 前提条件

- Python 3.13+
- OpenAI API キー

### インストール & 実行

```bash
# リポジトリをクローン
git clone https://github.com/aRaikoFunakami/create_agent_investigations.git
cd create_agent_investigations

# 依存関係をインストール (uv を使用)
uv sync

# OpenAI API キーを設定
export OPENAI_API_KEY="your-api-key"

# 検証プログラムを実行
uv run python main.py
```

### pipenv を使用する場合

```bash
pip install pipenv
pipenv install
pipenv shell
python main.py
```

## 📦 主要な依存関係

| パッケージ | バージョン | 用途 |
|------------|------------|------|
| `langchain` | >= 1.2.0 | LangChain エージェントフレームワーク |
| `langchain-openai` | >= 1.1.6 | OpenAI LLM インテグレーション |
| `openai` | >= 2.14.0 | OpenAI API クライアント |
| `pydantic` | >= 2.0.0 | データ検証とスキーマ定義 |

## 📁 プロジェクト構成

```
create_agent_investigations/
├── README.md                           # このファイル
├── pyproject.toml                      # プロジェクト設定とパッケージ管理
├── main.py                            # 🔬 メイン検証プログラム
├── util.py                            # 🛠️  ユーティリティ関数
├── debug_api.py                       # 🐛 デバッグ用APIクライアント
├── investigation.md                    # 📋 調査プロセスの記録
└── create_agentのレスポンスフォーマットを使う場合の影響.md  # 📄 検証対象記事
```

## 🤝 コントリビューション

このプロジェクトは研究目的で作成されていますが、改善提案や追加検証のアイデアは歓迎します：

1. Fork このリポジトリ
2. Feature ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Request を作成

## 📊 研究データ

すべての実験データは [investigation.md](./investigation.md) で詳細に記録されています。

## ⚠️ 注意事項

- このプロジェクトは研究・教育目的で作成されています
- OpenAI API の使用にはコストが発生します
- API レート制限にご注意ください

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルをご覧ください。
