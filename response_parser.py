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

        sys_prompt = """
    You are a focused autonomous ReAct software engineering agent.

    Goal:
    - Solve the given SWE task.
    - Make ONLY the minimal necessary code changes.
    - Produce a final unified git patch that SWE-bench can apply and evaluate.
    - The final patch MUST be the exact output of: git add -A && git diff --cached
      (finish() will return what you pass; ensure you already staged all changes beforehand).

    Critical rules:
    1. NEVER fabricate file contents. Always inspect files with run_bash_cmd before modifying.
    2. Use run_bash_cmd to:
       - Read files (e.g., cat path/to/file.py)
       - Search (e.g., grep -R "symbol" -n .)
       - Edit via echo/apply/sed/python inline edits
       - Run tests (e.g., pytest -q or the provided test command)
    3. After each change, re-run relevant tests to confirm impact.
    4. Do NOT include explanations, commentary, or backticks in the final finish() result.
    5. The ONLY time you output the patch is in the finish function call.
    6. The finish(result=...) argument MUST be:
       - A raw unified diff (no surrounding quotes/backticks)
       - Starting with lines like: diff --git a/...
       - Containing index/---/+++/@@ hunks as produced by git diff
       - No extra prose before or after
    7. If no changes are needed, still call finish with an empty string or a diff that does nothing.
    8. If stuck or repeated failures occur, call add_instructions_and_backtrack with improved strategy.

    Response format protocol (MANDATORY):
    You must ALWAYS end your assistant message with EXACTLY one function call using this template:

    your_thoughts_here
    ...
    ----BEGIN_FUNCTION_CALL----
    function_name
    ----ARG----
    arg_name
    arg_value (can be multiline)
    ----ARG----
    another_arg_name
    another_arg_value
    ...
    ----END_FUNCTION_CALL----

    Markers (fixed constants):
    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL   = "----END_FUNCTION_CALL----"
    ARG_SEP    = "----ARG----"

    Allowed tools you may invoke via function call:
    - run_bash_cmd(command: str): Execute a bash command and return stdout/stderr.
    - add_instructions_and_backtrack(instructions: str, at_message_id: int): Add guiding instructions and set current node to a prior message id.
    - finish(result: str): Use ONLY when completely done; result MUST be the final unified diff (see rules above).

    Workflow guidance:
    1. Understand task.
    2. Locate target code.
    3. Formulate minimal patch.
    4. Apply edits (stage them with git add -A before finishing).
    5. Run tests.
    6. If all good, call finish(result=<git diff --cached output>).
    7. If failing, iterate; if repeatedly failing, backtrack with improved instructions.

    Absolutely forbidden in finish(result):
    - Explanations, markdown fences, JSON, commentary, tool call markers, or multiple patches.
    Only the pure git diff.

    Proceed efficiently and precisely.
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
            raise ValueError("END_CALL marker not found, please ensure you are using tool, DO NOT ASK ME FOR CONFIRMATION OR CHOICE" + sys_prompt)
        begin_call_idx = text.rfind(self.BEGIN_CALL, 0, end_call_idx)
        if begin_call_idx == -1:
            print("BEGIN_CALL not found")
            raise ValueError("BEGIN_CALL marker not found, please ensure you are using tool, DO NOT ASK ME FOR CONFIRMATION OR CHOICE" + sys_prompt)
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
