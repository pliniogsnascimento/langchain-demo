import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Shared LLM — Haiku is fast and cheap, ideal for demos
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.7)


# ── Mode 1: Simple Chain ──────────────────────────────────────────────────────
def demo_simple_chain():
    """
    Demonstrates:
      - ChatPromptTemplate with variable placeholders
      - LCEL pipe operator (|) to compose a chain
      - StrOutputParser to unwrap AIMessage to a plain string
      - .stream() for token-by-token output
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful teacher. Explain concepts clearly and concisely."),
        ("human", "Explain '{concept}' in 3 bullet points."),
    ])

    # The | operator is LangChain Expression Language (LCEL)
    chain = prompt | llm | StrOutputParser()

    concept = input("Enter a concept to explain (e.g. 'recursion'): ").strip()
    print("\n── Response (streaming) ──")
    for chunk in chain.stream({"concept": concept}):
        print(chunk, end="", flush=True)
    print("\n")


# ── Mode 2: Conversation with Memory ─────────────────────────────────────────
def demo_conversation():
    """
    Demonstrates:
      - MessagesPlaceholder to inject chat history into a prompt
      - Manual history list (HumanMessage / AIMessage)
      - Multi-turn loop — Claude remembers prior context
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    history = []  # memory is just a list of messages

    print("Multi-turn chat — type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        response = chain.invoke({"input": user_input, "history": history})

        # Append to history so the next turn has full context
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))

        print(f"Claude: {response}\n")


# ── Mode 3: Agent with Tools ──────────────────────────────────────────────────
def demo_agent():
    """
    Demonstrates:
      - @tool decorator to expose Python functions to the LLM
      - create_tool_calling_agent (modern, no hub.pull required)
      - AgentExecutor with verbose=True to show the reasoning loop
    """

    @tool
    def calculator(expression: str) -> str:
        """Evaluates a mathematical expression. Input must be a valid Python math
        expression such as '2 + 2' or '100 * 1.08'."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_current_date(query: str = "") -> str:
        """Returns today's date. Use when the user asks about the current date."""
        return date.today().strftime("%B %d, %Y")

    tools = [calculator, get_current_date]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use tools when needed."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("Agent mode — ask something that needs calculation or today's date.")
    print("Example: 'What is a 15% tip on $47.80, and what is today's date?'\n")

    query = input("Your question: ").strip()
    print()
    result = executor.invoke({"input": query})
    print(f"\nFinal answer: {result['output']}")


# ── Main menu ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  LangChain + Claude Demo")
    print("=" * 50)
    print("1. Simple Chain  (PromptTemplate | LLM | Parser)")
    print("2. Conversation  (multi-turn with memory)")
    print("3. Agent         (tools: calculator + date)")
    print()

    choice = input("Pick a mode (1/2/3): ").strip()
    print()

    if choice == "1":
        demo_simple_chain()
    elif choice == "2":
        demo_conversation()
    elif choice == "3":
        demo_agent()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
