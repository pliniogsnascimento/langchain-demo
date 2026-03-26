import os
import operator
from typing import Annotated, List, TypedDict, Union

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

load_dotenv()

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    goal: str
    plan: List[str]
    code: str
    tests_passed: bool
    current_task: str

# --- LLM Setup ---
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

# --- Tool Definitions ---

@tool
def search_web(query: str):
    """Searches the web for information. Requires TAVILY_API_KEY."""
    # In a real scenario, use TavilySearchResults
    # For this demo, we'll return a mock response if the key is missing
    if not os.getenv("TAVILY_API_KEY"):
        return f"Searching for: {query}... (Mock response: Best practices for IDP involve using Backstage, Kubernetes, and Terraform.)"
    
    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults(max_results=3)
    return search.invoke(query)

@tool
def write_file(path: str, content: str):
    """Writes content to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"File {path} written successfully."

@tool
def run_command(command: str):
    """Runs a shell command and returns output."""
    import subprocess
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}
    except Exception as e:
        return str(e)

# --- Node Functions ---

def learner_node(state: AgentState):
    """Researches the topic based on the goal."""
    print("--- LEARNING ---")
    query = f"Internal Developer Platform best practices for: {state['goal']}"
    # Use the search tool (simplified here for brevity)
    info = search_web.invoke(query)
    return {"messages": [AIMessage(content=f"Research complete. Key findings: {info}")]}

def planner_node(state: AgentState):
    """Creates a plan of stories/tasks."""
    print("--- PLANNING ---")
    prompt = f"Based on the goal: {state['goal']}, create a simple list of tasks to implement it."
    response = llm.invoke(prompt)
    tasks = response.content.split("\n")
    return {"plan": tasks, "messages": [AIMessage(content=f"Plan created: {response.content}")]}

def implementer_node(state: AgentState):
    """Writes the code for the current task."""
    print("--- IMPLEMENTING ---")
    # In a real loop, we'd pick the next task from state['plan']
    task = state['plan'][0] if state['plan'] else state['goal']
    prompt = f"Implement the following task: {task}. Return only the python code."
    response = llm.invoke(prompt)
    state['code'] = response.content
    write_file.invoke({"path": "generated_platform_tool.py", "content": response.content})
    return {"code": response.content, "messages": [AIMessage(content="Implementation written to generated_platform_tool.py")]}

def tester_node(state: AgentState):
    """Generates and runs tests."""
    print("--- TESTING ---")
    test_code = f"import generated_platform_tool\ndef test_placeholder(): assert True"
    write_file.invoke({"path": "test_generated_tool.py", "content": test_code})
    
    # Run tests
    pytest_path = os.path.join(os.getcwd(), ".venv/bin/pytest")
    result = run_command.invoke(f"{pytest_path} test_generated_tool.py")
    passed = result['exit_code'] == 0
    return {"tests_passed": passed, "messages": [AIMessage(content=f"Tests passed: {passed}")]}

def publisher_node(state: AgentState):
    """Pushes to GitHub (Mocked)."""
    print("--- PUBLISHING ---")
    # result = run_command.invoke("git add . && git commit -m 'Add generated tool' && git push")
    return {"messages": [AIMessage(content="Code pushed to GitHub successfully (Simulated).")]}

# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("learner", learner_node)
workflow.add_node("planner", planner_node)
workflow.add_node("implementer", implementer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("publisher", publisher_node)

workflow.set_entry_point("learner")
workflow.add_edge("learner", "planner")
workflow.add_edge("planner", "implementer")
workflow.add_edge("implementer", "tester")

def route_after_test(state: AgentState):
    if state["tests_passed"]:
        return "publisher"
    else:
        return "implementer"

workflow.add_conditional_edges(
    "tester",
    route_after_test,
    {
        "publisher": "publisher",
        "implementer": "implementer"
    }
)

workflow.add_edge("publisher", END)

# Compile
app = workflow.compile()

# --- Main ---
if __name__ == "__main__":
    initial_state = {
        "goal": "Create a simple CLI tool that lists all S3 buckets (Mock implementation)",
        "messages": [],
        "plan": [],
        "code": "",
        "tests_passed": False,
        "current_task": ""
    }
    
    for event in app.stream(initial_state):
        for value in event.values():
            print(value["messages"][-1].content)
