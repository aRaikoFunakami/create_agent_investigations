"""
OpenAI API に実際に送信される内容を確認するデバッグプログラム
"""

import os
import json
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from util import ALL_TOOLS, SchemaWithValidation

# デバッグ用のパッチ
original_invoke = None

def debug_openai_call(self, input, config=None, **kwargs):
    """OpenAI API 呼び出しをインターセプトして内容を表示"""
    print("\n" + "="*60)
    print("🔍 OpenAI API への送信内容:")
    print("="*60)
    
    # ツール情報の表示
    if hasattr(self, '_tools') and self._tools:
        print(f"📋 Tools ({len(self._tools)}個):")
        for tool in self._tools:
            print(f"  - {tool.get('function', {}).get('name', 'unknown')}")
    
    # tool_choice の表示
    if hasattr(self, '_tool_choice'):
        print(f"⚙️ tool_choice: {self._tool_choice}")
    
    # メッセージの表示（最後のユーザーメッセージのみ）
    if isinstance(input, dict) and 'messages' in input:
        user_msg = None
        for msg in input['messages']:
            if hasattr(msg, 'type') and msg.type == 'human':
                user_msg = msg.content
        if user_msg:
            print(f"💬 User message: {user_msg}")
    
    print("-"*60)
    
    # 元の関数を呼び出し
    result = original_invoke(self, input, config, **kwargs)
    return result

def patch_openai():
    """OpenAI Chatモデルにパッチを当てる"""
    global original_invoke
    from langchain_openai.chat_models.base import ChatOpenAI
    if original_invoke is None:
        original_invoke = ChatOpenAI.invoke
        ChatOpenAI.invoke = debug_openai_call

def test_api_calls():
    """API呼び出しの内容をテスト"""
    patch_openai()
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    print("\n🧪 テスト1: response_format なし")
    agent_no_format = create_agent(llm, ALL_TOOLS, system_prompt="計算してください")
    try:
        agent_no_format.invoke({"messages": [("user", "10 + 5 を計算してください")]})
    except Exception as e:
        print(f"エラー（想定内）: {e}")
    
    print("\n🧪 テスト2: ToolStrategy")
    agent_tool = create_agent(
        llm, ALL_TOOLS, 
        response_format=ToolStrategy(schema=SchemaWithValidation),
        system_prompt="計算してください"
    )
    try:
        agent_tool.invoke({"messages": [("user", "10 + 5 を計算してください")]})
    except Exception as e:
        print(f"エラー（想定内）: {e}")
    
    print("\n🧪 テスト3: ProviderStrategy")
    agent_provider = create_agent(
        llm, ALL_TOOLS,
        response_format=ProviderStrategy(schema=SchemaWithValidation),
        system_prompt="計算してください"
    )
    try:
        agent_provider.invoke({"messages": [("user", "10 + 5 を計算してください")]})
    except Exception as e:
        print(f"エラー（想定内）: {e}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY が未設定")
    else:
        test_api_calls()