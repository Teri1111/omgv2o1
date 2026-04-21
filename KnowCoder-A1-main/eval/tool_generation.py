import re
import copy
import json
import os
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import random

from tool_env import ToolEnv, step, step_batch
from utils import *

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    generate_config: dict
    # max_start_length: int
    # max_prompt_length: int 
    # max_response_length: int
    # max_tool_response_length: int  # Renamed from max_obs_length
    # num_gpus: int
    # use_parallel_tool_calls: bool = False
    use_batch_tool_calls: bool = False  # New option for batch execution
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<tool_response>"
    tool_response_end: str = "</tool_response>"
    tool_custom_response_template: str = ""
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        llm_func, 
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.config = config
        self.is_validation = is_validation
        self.llm_func = llm_func

    def process_single_response(self, resp):
        # Look for tool call pattern: <tool_call>tool_name(args)</tool_call>
        tool_pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(tool_pattern, resp, re.DOTALL)
        
        if not match:
            # for custom tool call, can not generate <tool_call>...</tool_call>
            # pattern = r'({"name":\s*"[a-zA-Z0-9_]+",\s*"arguments":\s*{[^}]*}})'
            pattern = r'(\{"name":.*?\}\}\n)'
            match = re.search(pattern, resp)

            if match:
                tool_call_text = match.group(1)
                resp = resp.split('</think>')[0] + "</think>" + f"<tool_call>{tool_call_text}</tool_call>"
                return resp
            else:
                return resp  # No tool call found
        
        resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end
        # tool_content = match.group(0)
        
        # Replace all subsequent answer tag pairs with their content
        # rest_of_string = resp[match.end():]
        # cleaned_rest = re.sub(r'<tool_call>(.*?)</tool_call>', r'\1', rest_of_string, flags=re.DOTALL)
        
        return resp
    def _process_tool_call(self, responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            # Look for tool call pattern: <tool_call>tool_name(args)</tool_call>
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(tool_pattern, resp, re.DOTALL)
            
            if not match:
                return resp, False  # No tool call found
            
            resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end
            # tool_content = match.group(0)
            
            # Replace all subsequent answer tag pairs with their content
            # rest_of_string = resp[match.end():]
            # cleaned_rest = re.sub(r'<tool_call>(.*?)</tool_call>', r'\1', rest_of_string, flags=re.DOTALL)
            
            return resp, True
        
        # Process each response string
        return [process_single_response(resp)[0] for resp in responses_str], [process_single_response(resp)[1] for resp in responses_str]
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv]) -> List[str]:
        """Execute tool calls sequentially and return tool responses."""        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env) in enumerate(zip(response_strs, envs)):
            # Step the environment using the agent's response
            result = step(env, resp)
            tool_response = result[0]  # Extract observation from (observation, reward, done, info)
            tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
        return tool_responses

    def _extract_answer(self, response_str):
        # Look for \boxed{...} pattern within <answer>...</answer>
        answer_pattern = r'<answer>.*?\{([^}]*)\}.*?</answer>'
        match = re.search(answer_pattern, response_str, re.DOTALL)
        
        if not match:
            return None  # No boxed content found
        
        # Return the content inside \boxed{...}
        return match.group(1)
    def _verify_answer(self, prediction, golden_answers):
        def hit_check(prediction, golden_answers):
            score = 0.0
            if golden_answers == [] or not golden_answers:
                return 1.0
            if not isinstance(golden_answers, list):
                golden_answers = [golden_answers]
            try:
                prediction = eval(prediction)
            except:
                # print(prediction)
                return score
            if not isinstance(prediction, list):
                prediction = [prediction]
            for pred in prediction:
                if pred in golden_answers:
                    score = 1.0
                    break
            for answer in golden_answers:
                for pred in prediction:
                    if answer and type(answer) == str and type(pred) == str and answer in pred:
                        score = 1.0
                        break
                if score > 0.0:
                    break
            return score
        if hit_check(prediction, golden_answers) > 0:
            return True
        return False
    def run_llm_api_loop(self, start_message, envs: List[Any] = None, answers=None, mode='test') -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        # Main generation loop
        current_messages = copy.deepcopy(start_message)
        step = 0
        while step < self.config.max_turns:
            
            response = self.llm_func(current_messages, self.config.generate_config)
            if len(response) > 2000:
                response = response[:2000]  # Truncate response if too long
            if "<think<think" in response:
                response = response.replace("<think<think", '')
            if "<th<th" in response:
                response = response.replace("<th<th", '')
            if "<think><think>" in response:
                response = response.replace("<think><think>", '')
            print(colorful(f"LLM response @{step}: ", color="yellow"), end="")
            print(response)
            # temp, add the stop word '"}}\n' to the end of the response
            # stop_word = "</answer>"
            # if "<answer>" in response:
            #     if stop_word not in response:
            #         response += stop_word
            # elif "\"}}\n" not in response:
            #     response += "\"}}\n"
            responses_str = self.process_single_response(response)
            # idx = start_message[1]['content'].split('Question:')[-1].strip().split('Topic Entities')[0][:20]
            # print('********llm request consume: ', t1-t0)
            current_messages.append({
                "role": "assistant",
                "content": responses_str
            })
            # print(colorful(f"LLM response @{step}: ", color="yellow"), end="")
            # print(responses_str)

            # t0 = time.time()
            tool_response = self._execute_tool_calls([responses_str], envs)[0]
            # t1 = time.time()
            # print('********tool execution request consume: ', t1-t0)
            if '<answer>' in responses_str:
                current_messages.append({
                    "role": "user",
                    "content": tool_response
                })
                break
                # if not answers or (answers and  self._verify_answer(self._extract_answer(responses_str), answers)):
                #     tool_response = "You are right!"
                #     current_messages.append({
                #         "role": "user",
                #         "content": tool_response
                #     })
                #     # print(colorful("KB observation: ", color="yellow"), end="")
                #     # print(tool_response)
                #     break
                # elif mode == 'test':
                #     tool_response = "You are wrong!"
                #     current_messages.append({
                #         "role": "user",
                #         "content": tool_response
                #     })
                #     # print("extracted answer: ", self._extract_answer(responses_str))
                #     # print("gt answers: ", answers)
                #     break
                # else:
                #     # print("extracted answer: ", self._extract_answer(responses_str))
                #     # print("gt answers: ", answers)
                #     tool_response = self.config.tool_custom_response_template.format(tool_response="The answer is wrong or not well-formatted, retry your reasoning.")
            
            current_messages.append({
                "role": "user",
                "content": tool_response
            })
            # print(colorful("KB observation: ", color="yellow"), end="")
            # print(tool_response)
            # print(colorful(f"DEBUG: ", color="green"), end="")
            # print(current_messages[1:])
            # print(current_messages[-1])
            step += 1
        
        
        rollout_result = current_messages[1:]
        if rollout_result[0]['role'] == 'user':
            rollout_result = current_messages[2:]  # Skip the first user message if it exists
        return rollout_result
            
        

