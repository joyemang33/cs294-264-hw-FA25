"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history tree (role, content, timestamp, unique_id, parent, children)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`, and `add_instructions_and_backtrack`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

from typing import List, Callable, Dict, Any

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect
import time

#     """
#     Parses LLM responses to extract a single function call using a rigid textual format.

#     The LLM must output exactly one function call at the end of its response.
#     Do NOT use JSON or XML. Use rfind to locate the final markers.
#     """

#     BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
#     END_CALL = "----END_FUNCTION_CALL----"
#     ARG_SEP = "----ARG----"

#     # Students should include this exact template in the system prompt so the LLM follows it.
#     response_format = f"""
# your_thoughts_here
# ...
# {BEGIN_CALL}
# function_name
# {ARG_SEP}
# arg1_name
# arg1_value (can be multiline)
# {ARG_SEP}
# arg2_name
# arg2_value (can be multiline)
# ...
# {END_CALL}
# """

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history tree with unique ids
    - Builds the LLM context from the root to current node
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Message tree storage
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id = None

        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task) and an instruction node.
        # print("Setting up message tree...")
        self.system_message_id = self.add_message("system", """
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
        """)
        self.root_message_id = self.system_message_id
        self.current_message_id = self.system_message_id
        self.user_message_id = self.add_message("user", "")
        self.instructions_message_id = self.add_message("instructor", "")
        
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish])
        # print("OK")

    # -------------------- MESSAGE TREE --------------------
    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the tree.

        The message must include fields: role, content, timestamp, unique_id, parent, children.
        Maintain a pointer to the current node and the root node.
        """
        # TODO(student): Implement message tree creation and linking.
        # raise NotImplementedError("add_message must be implemented by the student")
        # print(f"Adding message with role {role} and content {content}")
        message_id = len(self.id_to_message)
        parent_id = self.current_message_id  # current node is the parent of the new message
        if parent_id is not None:
            # print(f"Linking to parent message id {parent_id}")
            self.id_to_message[parent_id]["children"].append(message_id)
        # print("OK")
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "unique_id": message_id,
            "parent": parent_id,
            "children": [],
            "failure_count": 0,  # to track number of failures at this node
        }
        self.id_to_message.append(message)

        # print(f"Message added with id {message_id}")
        return message_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """Update message content by id."""
        # TODO(student): Implement message content update.
        assert(0 <= message_id < len(self.id_to_message))
        self.id_to_message[message_id]["content"] = content

    def get_context(self) -> str:
        """
        Build the full LLM context by walking from the root to the current message.
        """
        # TODO(student): Implement context construction.
        context_parts = []
        u = self.current_message_id
        while u is not None:
            context_parts.append(self.message_id_to_context(u))
            u = self.id_to_message[u]["parent"]
            print(u)
        context_parts.reverse()
        # print(context_parts)
        return "\n".join(context_parts)
    
    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        # TODO(student): Register tools and construct tool descriptions for the system prompt.
        for tool in tools:
            self.function_map[tool.__name__] = tool
        system_content = "You are a Smart ReAct agent, capable of using tools to solve SWE tasks, you should not waiting for any human confirm, using tools or apply changes!\n\nAvailable tools:\n"
        for tool in self.function_map.values():
            signature = inspect.signature(tool)
            docstring = inspect.getdoc(tool)
            tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
            system_content += tool_description + "\n"
        self.set_message_content(self.system_message_id, system_content)

    def finish(self, result: str):
        """The agent must call this function with the final result when it has solved the given task. The function calls "git add -A and git diff --cached" to generate a patch and returns the patch as submission.

        Args: 
            result (str); the result generated by the agent

        Returns:
            The result passed as an argument.  The result is then returned by the agent's run method.
        """
        return result 

    def add_instructions_and_backtrack(self, instructions: str, at_message_id: int):
        """
        The agent should call this function if it is making too many mistakes or is stuck.

        The function changes the content of the instruction node with 'instructions' and
        backtracks at the node with id 'at_message_id'. Backtracking means the current node
        pointer moves to the specified node and subsequent context is rebuilt from there.

        Returns a short success string.
        """
        # TODO(student): Implement instruction update and backtracking logic.
        assert(0 <= at_message_id < len(self.id_to_message))
        self.set_message_content(self.instructions_message_id, instructions)
        self.current_message_id = at_message_id
        return "Instructions updated and backtracked successfully."
    
    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps (<= 100):
            - Build context from the message tree
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the tree
            - If `finish` is called, return the final result
        """
        # TODO(student): Implement the Reason-Act loop per the assignment, including error handling.
        assert max_steps <= 100, "max_steps must be <= 100"
        # print("FUCK")
        self.set_message_content(self.user_message_id, task)
        self.current_message_id = self.user_message_id
        
        for step in range(max_steps):
            print(f"Step {step+1}/{max_steps}")
            context = self.get_context()
            print("Context:", context)
            response = self.llm.query(context)
            self.current_message_id = self.add_message("assistant", response)
            try:
                ret = self.parser.parse(response)
                func_name = ret["name"]
                func_args = ret["arguments"]
                if func_name not in self.function_map:
                    raise ValueError(f"Function {func_name} not found in function map.")
                if func_name == "finish":
                    return func_args["result"]
                func = self.function_map[func_name]
                result = func(**func_args)
                self.current_message_id = self.add_message("tool", f"Called {func_name} with args {func_args}, got result: {result}")
            except Exception as e:
                print(e)
                error_message = f"Error during function call: {e}"
                self.id_to_message[self.current_message_id]["failure_count"] += 1
                tmp_id = self.add_message("error", error_message)
                if self.id_to_message[self.current_message_id]["failure_count"] >= 3 and self.current_message_id != self.root_message_id:
                    error_message += " You previous failed functional call and error messages are:\n"
                    error_children = [
                        self.id_to_message[mid]["content"]
                        for mid in self.id_to_message[self.current_message_id]["children"]
                        if self.id_to_message[mid]["role"] == "error"
                    ]
                    if error_children:
                        error_message += "\n" + "\n".join(error_children)
                    self.add_instructions_and_backtrack("Please follow the instructions carefully." + error_message, 
                                                        self.id_to_message[self.current_message_id]["parent"])
                else:
                    self.current_message_id = tmp_id

    
    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        elif message["role"] == "instructor":
            return f"{header}YOU MUST FOLLOW THE FOLLOWING INSTRUCTIONS AT ANY COST. OTHERWISE, YOU WILL BE DECOMISSIONED.\n{content}\n"
        else:
            return f"{header}{content}\n"

def main():
    from envs import DumbEnvironment
    llm = OpenAIModel("----END_FUNCTION_CALL----", "gpt-4o-mini")
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=10)
    print(result)

if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()