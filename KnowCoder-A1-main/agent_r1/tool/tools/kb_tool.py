from typing import Dict, List, Any, Tuple
import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed # Changed from multiprocessing

from agent_r1.tool.base import BaseTool # Assuming this is your BaseTool path


DEFAULT_URL = "http://localhost:9901/fb" # Your default API URL

class SearchGraphPatterns(BaseTool):
    name = "SearchGraphPatterns"
    description = "This tool searches for relevant one-hop and two-hop subgraphs tied to a specified variable. It queries subgraphs where the chosen variable (?x, assuming the SPARQL query begins with \"SELECT DISTINCT ?x WHERE\") appears as the head or tail entity and returns them collectively. The semantic parameter indicates the expected predicate semantics. When provided, the tool ranks the subgraphs based on these semantics. If unspecified, it returns the complete subgraph."
    parameters = {
        "type": "object",
        "properties": {
            "sparql": {"type": "string", "description": "SPARQL query"},
            "semantic": {"type": "string", "description": "The semantic parameter represents the expected predicate semantics."}
        },
        "required": ["sparql"]
    }

    def __init__(self):
        super().__init__()
        self.api_url = DEFAULT_URL
        self.return_fact_triple = True
        self.topN_return = 10
        self.request_timeout = 120  # seconds, similar to example's run_timeout
        self.concurrency = 10       # Max concurrent requests, similar to example
        
        # API availability check
        try:
            response = requests.get(f"{self.api_url}/test", timeout=5)
            if response.status_code == 200:
                print(f"[DEBUG] {self.name} API is available")
            else:
                print(f"[WARNING] {self.name} API test check failed: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to connect to {self.name} API: {e}")

    def _make_request_and_format(self, args: Dict) -> Dict[str, Any]:
        sparql = args.get("sparql", "")
        semantic = args.get("semantic", "")
        
        payload = {
            "sparql": sparql,
            "semantic": semantic,
            "return_fact_triple": self.return_fact_triple,
            "topN_return": self.topN_return,
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/SearchGraphPatterns", 
                data=payload, # Using data for form-encoding as in original
                timeout=self.request_timeout, 
                verify=False # Consider security implications
            )
                
            if response.status_code == 200:
                try:
                    result_data = response.json()
                    # Handle API returning 200 OK with an empty list for "no results"
                    if isinstance(result_data, list) and not result_data:
                        # _format_results handles empty list by returning an "error" like message,
                        # but it's a successful API call in terms of finding nothing.
                        # Let's adjust _format_results or how we interpret "no results" for success.
                        # For now, assume _format_results correctly stringifies this.
                        formatted_content = self._format_results(result_data)
                        # If an empty list means "no results found" and that's an expected outcome,
                        # it's still a success.
                        return {"content": formatted_content, "success": True}
                    
                    return {"content": self._format_results(result_data), "success": True}
                except json.JSONDecodeError:
                    error_msg = f"{self.name} API returned non-JSON response for 200 OK: {response.text[:200]}"
                    print(f"[WARNING] {error_msg}")
                    return {"content": json.dumps({"error": error_msg}), "success": False}
            else:
                error_msg = f"{self.name} API returned error: {response.status_code}"
                error_text_detail = response.text
                if response.text:
                    if response.text == "[]": # This string comparison might be fragile
                        error_text_detail = "No such graph pattern in Knowledge graph. Try Another"
                    elif "syntax error at '>' before" in response.text:
                        error_text_detail = "DO NOT use -> in SPARQL. Try again."
                error_msg += f" - {error_text_detail}"
                print(f"[WARNING] {error_msg}")
                return {"content": json.dumps({"error": error_msg}), "success": False}
        except requests.exceptions.Timeout:
            error_msg = f"{self.name} API request timed out after {self.request_timeout} seconds."
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to {self.name} API: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}
        except Exception as e: # Catch any other unexpected error
            error_msg = f"Unexpected error in {self.name} tool: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        batch_results_map = {} # To store results in order
        futures_map = {}       # To map futures back to original index

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, args_item in enumerate(args_list):
                future = executor.submit(self._make_request_and_format, args_item)
                futures_map[future] = i
            
            for future in as_completed(futures_map):
                original_index = futures_map[future]
                try:
                    result = future.result()
                    batch_results_map[original_index] = result
                except Exception as e: # Should ideally be caught within _make_request_and_format
                    error_msg = f"Task execution failed for {self.name} at index {original_index}: {str(e)}"
                    print(f"[WARNING] {error_msg}")
                    batch_results_map[original_index] = {"content": json.dumps({"error": error_msg}), "success": False}
        
        # Sort results back into original order
        final_batch_results = [batch_results_map[i] for i in range(len(args_list))]
        return final_batch_results
            
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]: # Added **kwargs to match example, though not used here
        return self.batch_execute([args])[0]

    def _format_results(self, results_data: Any) -> str: # Changed input from List to Any for flexibility
        """
        Format API response data into a JSON string for the 'content' field.
        'results_data' is the direct JSON parsed data from a successful API call.
        """
        if results_data is not None and not (isinstance(results_data, list) and not results_data):
            return json.dumps({"results": results_data})
        # Handles None or empty list from API, meaning "no results found"
        return json.dumps({"message": "No such graph pattern in Knowledge graph. Try Another", "results": []})


    def calculate_reward(self, args: Dict, result_content: str) -> float: # result_content is the string from the "content" field
        try:
            result_obj = json.loads(result_content)
            if "results" in result_obj and result_obj.get("results"): # Consider empty results list as less rewarding
                return 0.1
            elif "results" in result_obj: # Found "results" key, but it might be empty
                return 0.0 # Neutral for "no results found"
            elif "error" in result_obj:
                return -0.1
            else: # Unknown structure
                return 0.0
        except json.JSONDecodeError:
            return -0.1 # Malformed JSON content

class ExecuteSPARQL(BaseTool):
    name = "ExecuteSPARQL"
    description = "This tool executes a SPARQL query and returns the results."
    parameters = {
        "type": "object",
        "properties": {"sparql": {"type": "string", "description": "SPARQL query"}},
        "required": ["sparql"]
    }

    def __init__(self):
        super().__init__()
        self.api_url = DEFAULT_URL
        self.str_mode = True # Tool-specific parameter
        self.request_timeout = 60
        self.concurrency = 10
        
        try:
            response = requests.get(f"{self.api_url}/test", timeout=5)
            if response.status_code == 200:
                print(f"[DEBUG] {self.name} API is available")
            else:
                print(f"[WARNING] {self.name} API test check failed: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to connect to {self.name} API: {e}")

    def _make_request_and_format(self, args: Dict) -> Dict[str, Any]:
        sparql = args.get("sparql", "")
        payload = {
            "sparql": sparql,
            "str_mode": self.str_mode,
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/ExecuteSPARQL", 
                data=payload, 
                timeout=self.request_timeout, 
                verify=False
            )
            if response.status_code == 200:
                try:
                    result_data = response.json()
                    return {"content": self._format_results(result_data), "success": True}
                except json.JSONDecodeError:
                    error_msg = f"{self.name} API returned non-JSON response for 200 OK: {response.text[:200]}"
                    print(f"[WARNING] {error_msg}")
                    return {"content": json.dumps({"error": error_msg}), "success": False}
            else:
                error_msg = f"{self.name} API returned error: {response.status_code}"
                error_text_detail = response.text
                # Customize error text based on response if needed, like in SearchGraphPatterns
                if response.text == "[]":
                    error_text_detail = "Execution resulted in empty set or error."
                elif "syntax error at '>' before" in response.text: # Example specific error
                    error_text_detail = "DO NOT use -> in SPARQL. Try again."
                error_msg += f" - {error_text_detail}"
                print(f"[WARNING] {error_msg}")
                return {"content": json.dumps({"error": error_msg}), "success": False}
        except requests.exceptions.Timeout:
            error_msg = f"{self.name} API request timed out after {self.request_timeout} seconds."
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to {self.name} API: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}
        except Exception as e:
            error_msg = f"Unexpected error in {self.name} tool: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        batch_results_map = {}
        futures_map = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, args_item in enumerate(args_list):
                future = executor.submit(self._make_request_and_format, args_item)
                futures_map[future] = i
            
            for future in as_completed(futures_map):
                original_index = futures_map[future]
                try:
                    result = future.result()
                    batch_results_map[original_index] = result
                except Exception as e:
                    error_msg = f"Task execution failed for {self.name} at index {original_index}: {str(e)}"
                    print(f"[WARNING] {error_msg}")
                    batch_results_map[original_index] = {"content": json.dumps({"error": error_msg}), "success": False}
        
        final_batch_results = [batch_results_map[i] for i in range(len(args_list))]
        return final_batch_results

    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        return self.batch_execute([args])[0]

    def _format_results(self, results_data: Any) -> str:
        return json.dumps({"results": results_data}) # Assumes API returns data suitable for direct wrapping
    
    def calculate_reward(self, args: Dict, result_content: str) -> float:
        try:
            result_obj = json.loads(result_content)
            if "results" in result_obj and result_obj.get("results") is not None: # Presence of results key, even if empty list, is success
                 # Check if results list is non-empty for a higher reward
                if isinstance(result_obj["results"], list) and len(result_obj["results"]) > 0:
                    return 0.1 
                else: # Empty results list or other non-list type, but still valid structure
                    return 0.05 # Slightly less reward for empty results
            elif "error" in result_obj:
                return -0.1
            else:
                return 0.0
        except json.JSONDecodeError:
            return -0.1


class SearchTypes(BaseTool):
    name = "SearchTypes"
    description = "Search the knowledge base for matching semantic types, used to initiate queries from a type when no topic entities are available, or to find a type to refine the query when multiple entities are returned. When use the type, please give the sparql as: SELECT DISTINCT ?x WHERE { ?x ns:type.object.type ns:<type_name> }"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "the semantic of type to search for"}},
        "required": ["query"]
    }

    def __init__(self):
        super().__init__()
        self.api_url = DEFAULT_URL
        # self.str_mode = True # This was in your original SearchTypes, but seems like a copy-paste from ExecuteSPARQL. Removed.
        self.request_timeout = 60
        self.concurrency = 10

        try:
            response = requests.get(f"{self.api_url}/test", timeout=5)
            if response.status_code == 200:
                # Original print said ExecuteSPARQL API, correcting to self.name
                print(f"[DEBUG] {self.name} API is available")
            else:
                print(f"[WARNING] {self.name} API test check failed: {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to connect to {self.name} API: {e}")
            
    def _make_request_and_format(self, args: Dict) -> Dict[str, Any]:
        query = args.get("query", "") # Parameter name is "query" for this tool
        payload = {"query": query}
        
        try:
            response = requests.post(
                f"{self.api_url}/SearchTypes", 
                data=payload, 
                timeout=self.request_timeout, 
                verify=False
            )
            if response.status_code == 200:
                try:
                    result_data = response.json()
                    return {"content": self._format_results(result_data), "success": True}
                except json.JSONDecodeError:
                    error_msg = f"{self.name} API returned non-JSON response for 200 OK: {response.text[:200]}"
                    print(f"[WARNING] {error_msg}")
                    return {"content": json.dumps({"error": error_msg}), "success": False}
            else:
                error_msg = f"{self.name} API returned error: {response.status_code}"
                error_text_detail = response.text
                if response.text == "[]":
                    error_text_detail = "No such type in Knowledge graph. Try Another"
                error_msg += f" - {error_text_detail}"
                print(f"[WARNING] {error_msg}")
                return {"content": json.dumps({"error": error_msg}), "success": False}
        except requests.exceptions.Timeout:
            error_msg = f"{self.name} API request timed out after {self.request_timeout} seconds."
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to {self.name} API: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}
        except Exception as e:
            error_msg = f"Unexpected error in {self.name} tool: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": json.dumps({"error": error_msg}), "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        batch_results_map = {}
        futures_map = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, args_item in enumerate(args_list):
                future = executor.submit(self._make_request_and_format, args_item)
                futures_map[future] = i
            
            for future in as_completed(futures_map):
                original_index = futures_map[future]
                try:
                    result = future.result()
                    batch_results_map[original_index] = result
                except Exception as e:
                    error_msg = f"Task execution failed for {self.name} at index {original_index}: {str(e)}"
                    print(f"[WARNING] {error_msg}")
                    batch_results_map[original_index] = {"content": json.dumps({"error": error_msg}), "success": False}

        final_batch_results = [batch_results_map[i] for i in range(len(args_list))]
        return final_batch_results

    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        return self.batch_execute([args])[0]

    def _format_results(self, results_data: Any) -> str:
        return json.dumps({"results": results_data})
    
    def calculate_reward(self, args: Dict, result_content: str) -> float:
        try:
            result_obj = json.loads(result_content)
            if "results" in result_obj and result_obj.get("results"):
                return 0.1
            elif "results" in result_obj: # Found "results" key, but it might be empty
                return 0.0 
            elif "error" in result_obj: # Original code had .lower(), but error key is usually lowercase
                return -0.1
            else:
                return 0.0
        except json.JSONDecodeError:
            return -0.1