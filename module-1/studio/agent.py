from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def add(a: str, b: str) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    a = int(a)
    b = int(b)
    
    return a + b

def multiply(a: str, b: str) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    a = int(a)
    b = int(b)
    
    return a * b

def divide(a: str, b: str) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    a = int(a)
    b = int(b)
    
    return a / b

tools = [add, multiply, divide]

# Define LLM with bound tools
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
