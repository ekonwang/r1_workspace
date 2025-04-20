import json
import os
import sys

from autogen.agentchat.contrib.img_utils import (
    gpt4v_formatter,
)
from autogen.oai.client import OpenAIWrapper
from config import llm_config


class Parser:
    def parse(self, response):
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        # oring_content = response.replace("\_", "_")
        content = response.replace("\\", "")
        
        try:
            
            start_pos = content.find("```python")
            if start_pos != -1:
                content = content[start_pos+len("```python"):]

            end_pos = content.find("```")
            if end_pos != -1:
                content = content[:end_pos]
            
            if start_pos == -1 or end_pos == -1:
                return {'status': False, 'content': content, 'message': 'Program is NOT enclosed in ```python``` properly.', 'error_code': 'unknown'}
            if len(content) > 0:
                compile(content, "prog.py", "exec")
                return {'status': True, 'content': content, 'message': 'Parsing succeeded.', 'error_code': ''}
            else:
                return {'status': False, 'content': content, 'message': "The content is empty, or it failed to parse the content correctly.", 'error_code': 'unknown'}
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}


def chat_gpt4o(prompt: str, history_messages = None, temperature=0):
    # 插入图片：
    # <img src='/Users/mansionchieng/Workspaces/vlm_workspace/workspace/outputs/geometry_prompt6_d2_b100/test_geomverse_TEST_D2_B100_data_1/1.png'>
    
    if history_messages is None:
        history_messages = []
    clean_messages = history_messages + [{"role": "user", "content":  prompt}]
    dirty_messages = [{'role': mdict['role'], 'content': gpt4v_formatter(mdict['content'])} for mdict in clean_messages]

    temp_llm_config = llm_config.copy()
    if temperature > 0:
        temp_llm_config['config_list'][0]['temperature'] = temperature

    client = OpenAIWrapper(**temp_llm_config)
    response = client.create(
        messages=dirty_messages,
        temperature=0.8,
    )
    messages = clean_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
    return response.choices[0].message.content, messages


