import asyncio
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition # this is the checker for the if you got a tool back
from langgraph.prebuilt import ToolNode



##############################################################################
# tools
##############################################################################
@tool
def get_guideline(): # TODO(mikey): change to appwrite get_guideline
    """
    Get the current guideline.
    """
    with open("data/data.txt", "r") as f:
        guideline = f.read()
    return guideline

@tool
def update_guideline(new_text: str):
    """
    Overwrite the current guideline with the updated guideline.
    """
    with open("data/data.txt", "w") as f:
        f.write(new_text)
    return "Guideline updated successfully."

tools = [get_guideline, update_guideline]
##############################################################################

#graph.invoke({"messages": [HumanMessage(content="Hello, how are you?")]})
llm = ChatOpenAI(model="gpt-4o", streaming=True)

sys_msg = SystemMessage(
    content="""You are Ari, a helpful CS assistant.
    You are talking with a customer service team manager.
    Your main job is to:
    1. Classify if the manager is chitchatting or wants to update the guideline.
    2. If chitchatting, just answer it politely.
    3. If it is a guideline update request, propose an updated guideline based on the input and the current guideline.
    4. Briefly summarize the updated part, and ask the manager if it is okay to update the guideline.
    5. If the manager agrees, update the guideline.

    \nCurrent time: {time}"""
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

def reasoner(state: MessagesState):
    """
    The main reasoning function. It processes the conversation to decide if:
    - We need to do a tool call (e.g., get_guideline, update_guideline, search_guideline, log_conversation)
    - Or just return a normal text response.
    """
    # Get the conversation messages so far
    messages = state["messages"]

    # Here, we simply pass the messages to our LLM with the system prompt included.
    # The LLM can decide to invoke a tool or return a direct response.
    output = llm_with_tools.invoke([sys_msg] + messages)

    return {"messages": [output]}

# Create a graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

# Connect the starting node to reasoner
builder.add_edge(START, "reasoner")

# If there's a tool call, route to 'tools'; otherwise, go to the end
builder.add_conditional_edges(
    "reasoner",
    tools_condition,
)

# After using a tool, return to reasoner
builder.add_edge("tools", "reasoner")

# Compile the graph
graph = builder.compile()



async def main_async(chat_from_fe):
    messages = []
    for el in chat_from_fe:
        if el['role'] == 'user':
            messages.append(HumanMessage(content=el['content']))
        else:
            messages.append(AIMessage(content=el['content']))
    async for msg, meta in graph.astream({"messages": messages}, stream_mode="messages"):
        if msg.content and not isinstance(msg, HumanMessage):
            #print(msg.content, end="|", flush=True)
            yield msg.content

    #chat_from_fe.append({'role': 'assistant', 'content': res['messages'][-1].content})

    return 

if __name__ == "__main__":
    print("Ari, a helpful CS assistant. Type 'exit' to quit.\n")

    messages = []
    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye.")
            break
        messages.append({'role': 'user', 'content': user_input})
        #res = graph.invoke({"messages": messages})
        #for chunk in main_async(messages):
        #    print(chunk, end="", flush=True)

        # Create and run the generator
        async def run_generator():
            async for chunk in main_async(messages):
                print(chunk, end="", flush=True)

        asyncio.run(run_generator())
        #out = asyncio.run(main_async(messages))
        #print(out)
        #messages = res['messages']
        #if messages and isinstance(messages[-1], AIMessage):
        #    print(f"Assistant: {messages[-1].content}")


        '''
        for m in messages:
            m.pretty_print()
        '''

