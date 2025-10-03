class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
arg1_name
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
arg2_value (can be multiline)
...
{END_CALL}
"""

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        """
        # TODO(student): Implement rfind-based parsing per the assignment description.
        # Hints:
        # - Find END_CALL via rfind; then find the matching BEGIN_CALL before it via rfind
        # - Everything before BEGIN_CALL is the model's thought
        # - Between BEGIN_CALL and END_CALL: first block is function name, subsequent blocks
        #   are argument name/value pairs separated by ARG_SEP, values may be multiline
        # - Raise ValueError on malformed inputs
        # print("Parsing response:", text)
        end_call_idx = text.rfind(self.END_CALL)
        if end_call_idx == -1:
            print("END_CALL not found")
            raise ValueError("END_CALL marker not found")
        begin_call_idx = text.rfind(self.BEGIN_CALL, 0, end_call_idx)
        if begin_call_idx == -1:
            print("BEGIN_CALL not found")
            raise ValueError("BEGIN_CALL marker not found")
        thought = text[:begin_call_idx].strip()
        call_content = text[begin_call_idx + len(self.BEGIN_CALL):end_call_idx].strip()
        parts = call_content.split(self.ARG_SEP)
        if len(parts) < 1:
            print("No function name found")
            raise ValueError("No function name found")
        function_name = parts[0].strip()
        arguments = {}
        for i in range(1, len(parts)):
            # args and value are separeted by line
            arg_lines = parts[i].strip().split("\n", 1)
            if len(arg_lines) != 2:
                print(f"Malformed argument block: {parts[i]}")
                raise ValueError(f"Malformed argument block: {parts[i]}")
            arg_name = arg_lines[0].strip()
            arg_value = arg_lines[1].strip()
            arguments[arg_name] = arg_value

        return {"thought": thought, "name": function_name, "arguments": arguments}
