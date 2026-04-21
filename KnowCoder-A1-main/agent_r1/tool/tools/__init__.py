def _default_tool(name):
    if name == "search":
        from agent_r1.tool.tools.search_tool import SearchTool
        return SearchTool()
    elif name == "wiki_search":
        from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
        return WikiSearchTool()
    elif name == "python":
        from agent_r1.tool.tools.python_tool import PythonTool
        return PythonTool()
    elif name == 'SearchGraphPatterns':
        from agent_r1.tool.tools.kb_tool import SearchGraphPatterns
        return SearchGraphPatterns()
    elif name == 'SearchTypes':
        from agent_r1.tool.tools.kb_tool import SearchTypes
        return SearchTypes()
    elif name == 'ExecuteSPARQL':
        from agent_r1.tool.tools.kb_tool import ExecuteSPARQL
        return ExecuteSPARQL()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")
    