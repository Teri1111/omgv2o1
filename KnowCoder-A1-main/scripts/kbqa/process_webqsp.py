
prompt_template = """#Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{'type': 'function', 'function': {'name': 'SearchGraphPatterns', 'description': 'This tool searches for relevant one-hop and two-hop subgraphs tied to a specified variable. It queries subgraphs where the chosen variable (?x, assuming the SPARQL query begins with "SELECT DISTINCT ?x WHERE") appears as the head or tail entity and returns them collectively. The semantic parameter indicates the expected predicate semantics. When provided, the tool ranks the subgraphs based on these semantics. If unspecified, it returns the complete subgraph.', 'parameters': {'type': 'object', 'properties': {'sparql': {'type': 'string', 'description': 'SPARQL query'}, 'semantic': {'type': 'string', 'description': 'The semantic parameter represents the expected predicate semantics.'}}, 'required': ['sparql']}}}
{'type': 'function', 'function': {'name': 'ExecuteSPARQL', 'description': 'This tool executes a SPARQL query and returns the results.', 'parameters': {'type': 'object', 'properties': {'sparql': {'type': 'string', 'description': 'SPARQL query'}}, 'required': ['sparql']}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
Answer the given question. You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>

Example:
Question: What timezone is New Mexico currently in?
Topic Entities: New Mexico(m.05fjy)
assistant:
<think>
To answer, I need to identify New Mexico’s timezone using SPARQL. New Mexico’s entity ID is m.05fjy, and I need a predicate linking it to a timezone. I’ll use SearchGraphPatterns to explore relations for New Mexico (?x0). The SPARQL query starts with SELECT DISTINCT ?x WHERE { VALUES ?x0 {ns:m.05fjy}}, with the semantic parameter set to "timezone". 
</think>
<tool_call>
{{"name": SearchGraphPatterns, "arguments": {"sparql": "SELECT DISTINCT ?x WHERE { VALUES ?x {ns:m.05fjy}}","semantic": "time zone"}}}
</tool_call>
user:
<tool_response>
{'results': '[("Mountain Time Zone", time.time_zone.locations_in_this_time_zone, ?x), (?x, location.location.time_zones, "Mountain Time Zone"), ("2012 Filmapalooza", time.event.locations, ?x), (?x, location.location.area, "315194.0"^^xsd:float), (?x, location.dated_location.date_founded, "1912-01-06"^^xsd:date)]'}
</tool_response>
assistant:
<think>
The SearchGraphPatterns tool showed two relevant relations: one linking the Mountain Time Zone to locations (time.time_zone.locations_in_this_time_zone) and another linking New Mexico to its timezone (location.location.time_zones). The latter is more direct since we want New Mexico's timezone.
The SPARQL query should select the timezone for New Mexico using the location.location.time_zones predicate.
</think>
<tool_call>
{{"name": ExecuteSPARQL, "arguments": {"sparql": "SELECT DISTINCT ?x1 WHERE {\nVALUES ?x0{ns:m.05fjy} #ns:m.05fjy is New Mexico\n?x0 ns:location.location.time_zones ?x1. # Mountain Time Zone\n}"}}}
</tool_call>
user:
<tool_response>
{'results': '["Mountain Time Zone"]'}
</tool_response>
assistant:
<think>
That's the sparql to get the answer.
</think>
<answer>
["Mountain Time Zone"]
</answer>

Question: What character did John Noble play in The Lord of the Rings?
Topic Entities: John Noble (m.02fgm7), The Lord of the Rings: The Two Towers (m.017gm7)"""