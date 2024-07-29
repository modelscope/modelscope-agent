CYPHER_PROMPT = """[start_of_cypher_queries]
### Query 1
**decomposed text query**:
```cypher
<cypher_query>
```

### Query 2
**decomposed text query**:
```cypher
<cypher_query>
```
...

### Query n
**decomposed text query**:
```cypher
<cypher_query>
```
[end_of_cypher_queries]
"""

JSON_PROMPT = """```json
{{"thought": $THOUGHT, "action": $ACTION_NAME, "action_input": $INPUT}}
```
"""
