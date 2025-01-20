import re

def extract_first_function(code: str) -> str:
    # Regular expression to match the first function definition and its body
    function_pattern = re.compile(r'(def\s+\w+\s*\(.*?\)\s*:\s*(?:\n\s+.*)+)', re.DOTALL)
    
    # Search for the first function definition
    match = function_pattern.search(code)
    
    if match:
        # Extract the function body
        function_body = match.group(1).strip()
        return function_body
    else:
        return "No function found in the provided code."

# Example usage
code = """
invalid code here

def example_function(x):
    y = x + 1
    return y

more invalid code here

print(example_function(5))
"""

function_body = extract_first_function(code)

print(f"Extracted function:\n{function_body}")
