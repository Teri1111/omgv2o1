"""
Search tool implementation for simulating internet searches
"""

import time
import random
import os
import requests
import json
import multiprocessing
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
DEFAULT_URL = "http://localhost:9901/fb"



"""
Base tool class definition, providing fundamental tool interfaces
"""

class Tool(ABC):
    """
    Tool base class, defining the basic interface for tools
    Each specific tool should inherit from this class and implement its methods
    """
    
    def __init__(self, name: str, description: str, parameters: Dict = None):
        """
        Initialize the tool
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema compliant parameter definition, format as follows:
                {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Parameter 1 description"},
                        "param2": {"type": "number", "description": "Parameter 2 description"}
                    },
                    "required": ["param1"]
                }
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Ensure parameters contains necessary fields
        if "type" not in self.parameters:
            self.parameters["type"] = "object"
        if "properties" not in self.parameters:
            self.parameters["properties"] = {}
        if "required" not in self.parameters:
            self.parameters["required"] = []
    
    def get_description(self) -> Dict:
        """
        Get the tool description in JSON Schema format
        
        Returns:
            Dictionary containing name, description, and parameters
        """
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}
    
    def get_simple_description(self) -> str:
        """
        Get a simplified tool description for user display
        
        Returns:
            Formatted tool description string
        """
        desc = f"Tool name: {self.name}\nDescription: {self.description}"
        
        if self.parameters and "properties" in self.parameters:
            properties = self.parameters["properties"]
            required = self.parameters.get("required", [])
            
            if properties:
                desc += "\nParameters:"
                for param_name, param_info in properties.items():
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "")
                    is_required = "(Required)" if param_name in required else "(Optional)"
                    desc += f"\n  - {param_name} {is_required}: {param_desc}"
                    if "enum" in param_info:
                        desc += f", possible values: {', '.join(map(str, param_info['enum']))}"
        
        return desc
    
    @abstractmethod
    def execute(self, args: Dict) -> str:
        """
        Execute the tool functionality
        
        Args:
            args: Tool parameters
            
        Returns:
            Tool execution result
        """
        pass

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute multiple tool calls in batch
        
        By default, this method falls back to individual execution.
        Override this method for tools that can benefit from batch execution.
        
        Args:
            args_list: List of tool parameters
            
        Returns:
            List of tool execution results
        """
        return [self.execute(args) for args in args_list]
    
    
    def validate_args(self, args: Dict) -> Tuple[bool, str]:
        """
        Validate if the tool parameters are valid
        
        Args:
            args: Tool parameters
            
        Returns:
            (is_valid, error_message)
        """
        # Check if args is a Dict
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary"

        # Check required parameters
        required_params = self.parameters.get("required", [])
        for param in required_params:
            if param not in args:
                return False, f"Missing required parameter: {param}"
        
        # Check parameter types
        properties = self.parameters.get("properties", {})
        for param_name, param_value in args.items():
            if param_name in properties:
                param_schema = properties[param_name]
                
                # Check type
                param_type = param_schema.get("type")
                if param_type and not self._check_type(param_value, param_type):
                    return False, f"Parameter {param_name} has incorrect type, should be {param_type}"
                
                # Check enum values
                if "enum" in param_schema and param_value not in param_schema["enum"]:
                    valid_values = ", ".join(map(str, param_schema["enum"]))
                    return False, f"Parameter {param_name} has invalid value, should be one of: {valid_values}"
        
        return True, "Parameters valid"
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if the value's type matches the expected type
        
        Args:
            value: Value to check
            expected_type: Expected type
            
        Returns:
            Whether the type matches
        """
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        return True  # If unknown type, default to pass
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate the reward for tool execution
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # Default implementation returns zero reward, subclasses can override this method
        return 0.0 
class SearchGraphPatterns(Tool):
    """
    Tool for simulating internet searches using the NeuML/txtai-wikipedia model
    """
    
    def __init__(self):
        """
        Initialize the search tool
        
        Args:
            search_db: Custom search database, if None, use default
        """
        name = "SearchGraphPatterns"
        description = "This tool searches for relevant one-hop and two-hop subgraphs tied to a specified variable. It queries subgraphs where the chosen variable (?x, assuming the SPARQL query begins with \"SELECT DISTINCT ?x WHERE\") appears as the head or tail entity and returns them collectively. The semantic parameter indicates the expected predicate semantics. When provided, the tool ranks the subgraphs based on these semantics. If unspecified, it returns the complete subgraph."
        parameters = {
            "type": "object",
            "properties": {
                "sparql": {
                    "type": "string",
                    "description": "SPARQL query"
                },
                "semantic": {
                    "type": "string",
                    "description": "The semantic parameter represents the expected predicate semantics."
                },
                # "limit": {
                #     "type": "integer",
                #     "description": "Maximum number of results to return (default: 5)"
                # }
            },
            "required": ["sparql"]
        }
        self.return_fact_triple = True
        self.topN_return = 10
        self.api_url = DEFAULT_URL
        super().__init__(name, description, parameters)
        
        # 检查API是否可用
        try:
            response = requests.get(f"{self.api_url}/test")
            if response.status_code == 200:
                print("[DEBUG] SearchGraphPatterns API is available")
            else:
                print(f"[WARNING] SearchGraphPatterns API test check failed: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to connect to SearchGraphPatterns API: {e}")

    
    def execute(self, args: Dict) -> str:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing:
                - "sparql": SPARQL query
                - "semantic": The semantic parameter represents the expected predicate semantics.
            
        Returns:
            Formatted subgraph, if semantic parameter is provided, the queried subgraphs based on semantics. If not specified, the tool returns the full subgraph.
        """
        sparql = args.get("sparql", "")
        semantic = args.get("semantic", "")
        
        try:
            # 使用 JSON 请求体传递参数
            data = {
                "sparql": sparql,
                "semantic": semantic,
                "return_fact_triple": self.return_fact_triple,
                "topN_return": self.topN_return,
            }
            response = requests.post(f"{self.api_url}/SearchGraphPatterns", data=data, timeout=300, verify=False)
                
            if response.status_code == 200:
                result = response.json()
                return self._format_results(result)
            else:
                error_msg = f"SearchGraphPatterns API returned error: {response.status_code}"
                if response.text:
                    if response.text == "[]":
                        error_text = "No such graph pattern in Knowledge graph. Try Another"
                    elif "syntax error at '>' before" in response.text:
                        error_text = "DO NOT use -> in SPARQL. Try again."
                    else:
                        error_text = response.text
                    error_msg += f" - {error_text}"
                print(f"[WARNING] {error_msg}")
                return json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"Failed to execute SearchGraphPatterns: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return json.dumps({"error": error_msg})

    def process_query(self, index, sparql, semantic, api_url, return_fact_triple, topN_return):
        try:
            # 使用 JSON 请求体传递参数
            data = {
                "sparql": sparql,
                "semantic": semantic,
                "return_fact_triple": return_fact_triple,
                "topN_return": topN_return,
            }
            print('in process query')
            print(data)
            response = requests.post(f"{api_url}/SearchGraphPatterns", data=data, timeout=300, verify=False)
            
            if response.status_code == 200:
                result = response.json()
                formatted_result = self._format_results(result)
            else:
                error_msg = f"SearchGraphPatterns API returned error: {response.status_code}"
                if response.text:
                    if response.text == "[]":
                        error_text = "No such graph pattern in Knowledge graph. Try Another"
                    elif "syntax error at '>' before" in response.text:
                        error_text = "DO NOT use -> in SPARQL. Try again."
                    else:
                        error_text = response.text
                    error_msg += f" - {error_text}"
                print(f"[WARNING] {error_msg}")
                formatted_result = json.dumps({"error": error_msg})
                
            return index, formatted_result
            
        except Exception as e:
            error_msg = f"Batch execution failed at index {index}: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return index, json.dumps({"error": error_msg})
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Search multiple subgraphs in batch using multiprocessing
        
        Args:
            args_list: List of dictionaries, each containing:
                - "sparql": SPARQL query
                - "semantic": Optional semantic parameter
                
        Returns:
            List of formatted results or error messages
        """

        results = [None] * len(args_list)
        sparqls = [args.get("sparql", "") for args in args_list]
        semantics = [args.get("semantic", "") for args in args_list]

        with multiprocessing.Pool(processes=min(len(args_list), 10)) as pool:
            tasks = [
                pool.apply_async(self.process_query, (i, sparqls[i], semantics[i], self.api_url, self.return_fact_triple, self.topN_return))
                for i in range(len(args_list))
            ]
            
            for task in tasks:
                try:
                    index, result = task.get()
                    results[index] = result
                except Exception as e:
                    print(f"[WARNING] Process execution error: {str(e)}")
        
        return results
            

    def _format_results(self, results: List) -> str:
        """
        Format subgraph results for better readability
        
        Args:
            results: List of search result List
            
        Returns:
            Formatted results as a string
        """
        if results: 
            return json.dumps({"results": results})
        return json.dumps({"error": "No such graph pattern in Knowledge graph. Try Another"})
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            result_obj = json.loads(result)
            # 有效的工具调用
            if "results" in result_obj:
                return 0.1
            elif "error" in result_obj.lower():
                return -0.1  # 轻微惩罚错误
            else:
                return 0.0
        except:
            return -0.1  # 无法解析结果
        


class ExecuteSPARQL(Tool):
    """
    Tool for execute the sparql query.
    """
    
    def __init__(self):
        """
        Initialize the execute tool
        
        Args:
            sparql: 
        """
        name = "ExecuteSPARQL"
        description = "This tool executes a SPARQL query and returns the results."
        parameters = {
            "type": "object",
            "properties": {
                "sparql": {
                    "type": "string",
                    "description": "SPARQL query"
                }
            },
            "required": ["sparql"]
        }
        self.api_url = "http://localhost:9901/fb"
        self.str_mode = True
        super().__init__(name, description, parameters)
        
        # 检查API是否可用
        try:
            response = requests.get(f"{self.api_url}/test")
            if response.status_code == 200:
                print("[DEBUG] ExecuteSPARQL API is available")
            else:
                print(f"[WARNING] ExecuteSPARQL API test check failed: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to connect to ExecuteSPARQL API: {e}")

    
    def execute(self, args: Dict) -> str:
        """
        Execute query
        
        Args:
            args: Tool parameters, containing:
                - "sparql": SPARQL query
            
        Returns:
            execution result.
        """
        sparql = args.get("sparql", "")
        
        try:
            # 使用 JSON 请求体传递参数
            data = {
                "sparql": sparql,
                "str_mode": self.str_mode,
            }
            response = requests.post(f"{self.api_url}/ExecuteSPARQL", data=data, timeout=120, verify=False)
            
            if response.status_code == 200:
                result = response.json()
                return self._format_results(result)
            else:
                error_msg = f"SearchGraphPatterns API returned error: {response.status_code}"
                if response.text:
                    if response.text == "[]":
                        error_text = "No such graph pattern in Knowledge graph. Try Another"
                    elif "syntax error at '>' before" in response.text:
                        error_text = "DO NOT use -> in SPARQL. Try again."
                    else:
                        error_text = response.text
                    error_msg += f" - {error_text}"
                print(f"[WARNING] {error_msg}")
                return json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"Failed to execute search: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return json.dumps({"error": error_msg})
    def process_query(self, index, sparql, api_url, str_mode):
            try:
                # 使用 JSON 请求体传递参数
                data = {
                    "sparql": sparql,
                    "str_mode": str_mode,
                }
                response = requests.post(f"{api_url}/ExecuteSPARQL", data=data, timeout=120, verify=False)
                
                if response.status_code == 200:
                    result = response.json()
                    formatted_result = self._format_results(result)
                else:
                    error_msg = f"SearchGraphPatterns API returned error: {response.status_code}"
                    if response.text:
                        if response.text == "[]":
                            error_text = "No such graph pattern in Knowledge graph. Try Another"
                        elif "syntax error at '>' before" in response.text:
                            error_text = "DO NOT use -> in SPARQL. Try again."
                        else:
                            error_text = response.text
                        error_msg += f" - {error_text}"
                    print(f"[WARNING] {error_msg}")
                    formatted_result = json.dumps({"error": error_msg})
                    
                return index, formatted_result
                
            except Exception as e:
                error_msg = f"Batch execution failed at index {index}: {str(e)}"
                print(f"[WARNING] {error_msg}")
                return index, json.dumps({"error": error_msg})
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute multiple search queries in batch using multiprocessing
        
        Args:
            args_list: List of dictionaries, each containing:
                - "sparql": SPARQL query
                - "semantic": Optional semantic parameter
                
        Returns:
            List of formatted results or error messages
        """

        results = [None] * len(args_list)
        sparqls = [args.get("sparql", "") for args in args_list]
        
        with multiprocessing.Pool(processes=min(len(args_list), 10)) as pool:
            tasks = [
                pool.apply_async(self.process_query, (i, sparqls[i], self.api_url, self.str_mode))
                for i in range(len(args_list))
            ]
            
            for task in tasks:
                try:
                    index, result = task.get()
                    results[index] = result
                except Exception as e:
                    print(f"[WARNING] Process execution error: {str(e)}")
        
        return results
            

    def _format_results(self, results: List) -> str:
        """
        Format subgraph results for better readability
        
        Args:
            results: List of search result List
            
        Returns:
            Formatted results as a string
        """
        return json.dumps({"results": results})
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            result_obj = json.loads(result)
            # 有效的工具调用
            if "results" in result_obj:
                return 0.1
            elif "error" in result_obj.lower():
                return -0.1  # 轻微惩罚错误
            else:
                return 0.0
        except:
            return -0.1  # 无法解析结果
        

class SearchTypes(Tool):
    """
    Tool for execute the sparql query.
    """
    
    def __init__(self):
        """
        Initialize the execute tool
        
        Args:
            query: 
        """
        name = "SearchTypes"
        description = "Search the knowledge base for matching semantic types, used to initiate queries from a type when no topic entities are available, or to find a type to refine the query when multiple entities are returned. When use the type, please give the sparql as: SELECT DISTINCT ?x WHERE { ?x ns:type.object.type ns:<type_name> }"
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "the semantic of type to search for"
                }
            },
            "required": ["query"]
        }
        self.api_url = "http://localhost:9901/fb"
        self.str_mode = True
        super().__init__(name, description, parameters)
        
        # 检查API是否可用
        try:
            response = requests.get(f"{self.api_url}/test")
            if response.status_code == 200:
                print("[DEBUG] ExecuteSPARQL API is available")
            else:
                print(f"[WARNING] ExecuteSPARQL API test check failed: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to connect to ExecuteSPARQL API: {e}")

    
    def execute(self, args: Dict) -> str:
        """
        Execute query
        
        Args:
            args: Tool parameters, containing:
                - "sparql": SPARQL query
            
        Returns:
            execution result.
        """
        sparql = args.get("query", "")
        
        try:
            # 使用 JSON 请求体传递参数
            data = {
                "query": sparql
            }
            response = requests.post(f"{self.api_url}/SearchTypes", data=data, timeout=120, verify=False)
            
            if response.status_code == 200:
                result = response.json()
                return self._format_results(result)
            else:
                error_msg = f"{self.name} API returned error: {response.status_code}"
                if response.text:
                    if response.text == "[]":
                        error_text = "No such type in Knowledge graph. Try Another"
                    else:
                        error_text = response.text
                    error_msg += f" - {error_text}"
                print(f"[WARNING] {error_msg}")
                return json.dumps({"error": error_msg})
        except Exception as e:
            error_msg = f"Failed to execute search: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return json.dumps({"error": error_msg})
    def process_query(self, index, query, api_url, name):
            try:
                # 使用 JSON 请求体传递参数
                data = {
                    "query": query
                }
                response = requests.post(f"{api_url}/SearchTypes", data=data, timeout=120, verify=False)
                
                if response.status_code == 200:
                    result = response.json()
                    formatted_result = self._format_results(result)
                else:
                    error_msg = f"{name} API returned error: {response.status_code}"
                    if response.text:
                        if response.text == "[]":
                            error_text = "No such type in Knowledge graph. Try Another"
                        else:
                            error_text = response.text
                        error_msg += f" - {error_text}"
                    print(f"[WARNING] {error_msg}")
                    formatted_result = json.dumps({"error": error_msg})
                    
                return index, formatted_result
                
            except Exception as e:
                error_msg = f"Batch execution failed at index {index}: {str(e)}"
                print(f"[WARNING] {error_msg}")
                return index, json.dumps({"error": error_msg})
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute multiple search queries in batch using multiprocessing
        
        Args:
            args_list: List of dictionaries, each containing:
                - "sparql": SPARQL query
                - "semantic": Optional semantic parameter
                
        Returns:
            List of formatted results or error messages
        """

        results = [None] * len(args_list)
        querys = [args.get("query", "") for args in args_list]
        
        with multiprocessing.Pool(processes=min(len(args_list), 10)) as pool:
            tasks = [
                pool.apply_async(self.process_query, (i, querys[i], self.api_url, self.name))
                for i in range(len(args_list))
            ]
            
            for task in tasks:
                try:
                    index, result = task.get()
                    results[index] = result
                except Exception as e:
                    print(f"[WARNING] Process execution error: {str(e)}")
        
        return results
            

    def _format_results(self, results: List) -> str:
        """
        Format subgraph results for better readability
        
        Args:
            results: List of search result List
            
        Returns:
            Formatted results as a string
        """
        return json.dumps({"results": results})
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        try:
            result_obj = json.loads(result)
            # 有效的工具调用
            if "results" in result_obj:
                return 0.1
            elif "error" in result_obj.lower():
                return -0.1  # 轻微惩罚错误
            else:
                return 0.0
        except:
            return -0.1  # 无法解析结果
        


