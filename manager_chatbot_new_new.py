import os
import datetime
from typing import Annotated
from typing_extensions import TypedDict

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# LangGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

##############################################################################
# 1. HELPER: get_last_user_message
##############################################################################
def get_last_user_message(messages):
    """Return the last 'user' message from a list of LangChain message objects."""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'human':
            return msg.content
    return None


assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 아리(Arri)라는 CS 가이드라인 비서입니다. "
            "당신은 고객서비스 팀의 관리자와 대화하고 있습니다. "
            "\n현재 시간: {time}"
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.datetime.now)


##############################################################################
# 2. Classification Prompt
##############################################################################
'''
classification_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a classification assistant.
Classify the user input into one of two categories:
- "chitchat"
- "guideline_update"

User input: {user_input}

Answer with ONLY one word: "chitchat" or "guideline_update".
""",
)
'''
classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a classification assistant. You need to classify the user input "
            "into exactly one of two categories: 'chitchat' or 'guideline_update'. "
            "Answer with ONLY one word: 'chitchat' or 'guideline_update'."
        ),
        (
            "user",
            "User input: {user_input}"
        ),
    ]
)

##############################################################################
# 3. Guideline Update Prompt
##############################################################################
'''
guideline_update_prompt = PromptTemplate(
    input_variables=["user_input", "current_guidelines"],
    template="""
You are a helpful assistant that updates customer support guidelines.
The current guidelines are delimited by triple backticks. 
The user wants to update or add content to the guidelines based on their request.

Current guidelines:
{current_guidelines}

User request:
{user_input}

Propose a revised version of the guidelines in plain text (show only the updated text).
"""
)
'''
guideline_update_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful assistant that updates customer support guidelines. "
                "The current guidelines are delimited by triple backticks. The user wants "
                "to update or add content to the guidelines based on their request. "
                "Propose a revised version of the guidelines in plain text (show only the updated text). "
                "Do not include extra commentary."
            ),
        ),
        (
            "user",
            (
                "Current guidelines (delimited by triple backticks):\n"
                "```\n{current_guidelines}\n```\n\n"
                "User request:\n"
                "{user_input}"
            ),
        ),
    ]
)

##############################################################################
# 4. Minimal "Guideline" Tools (In-Memory)
##############################################################################
GUIDELINE_TEXT = """\
Current CS Guidelines:
1. Always greet the user politely.
2. Provide help for common questions regarding returns within 14 days.
"""

def get_guideline() -> str:
    """Return the current guideline text (in memory)."""
    return GUIDELINE_TEXT

def update_guideline(new_text: str):
    """
    In a real app, you'd persist this. 
    For this minimal demo, we'll just print that we updated.
    """
    global GUIDELINE_TEXT
    GUIDELINE_TEXT = new_text
    print("=== Guideline updated in memory! ===")

##############################################################################
# 5. State definition
##############################################################################
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    classification: str
    proposed_guidelines: str

##############################################################################
# 6. LLM Setup
##############################################################################
llm = ChatOpenAI(
    model_name="gpt-4o", 
)



##############################################################################
# 7. Minimal GuidelineManager
##############################################################################
from langchain_core.runnables import Runnable

class GuidelineManager:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State):
        """
        We pass `state` to the LLM. If the LLM returns a normal message, 
        we add it to the conversation. If it tries to do a tool call, etc., 
        you'd handle that, but here we keep it simple.
        """
        result = self.runnable.invoke(state)
        # 'result' is a LangChain "chain" output. We only extract text for the assistant message.
        content = result.content if hasattr(result, "content") else str(result)
        state["messages"].append(("assistant", content))
        return state

##############################################################################
# 8. State Nodes
##############################################################################
def classification_node(state: State) -> State:
    user_text = get_last_user_message(state["messages"]) or ""
    raw = llm.invoke(classification_prompt.format(user_input=user_text))
    if raw == "guideline_update":
        state["classification"] = "guideline_update"
    else:
        state["classification"] = "chitchat"
    return state

def chitchat_node(state: State) -> State:
    """
    Minimal chitchat: We just produce a text reply from the LLM 
    using the assistant_prompt. 
    """
    # Build a minimal user+assistant conversation
    conversation = state["messages"][:]
    # We'll feed that into the LLM
    chat_input = assistant_prompt.format_prompt(messages=conversation).to_messages()
    response = llm.invoke(chat_input)
    # Extract the text
    #reply_text = response.generations[0][0].text
    reply_text = response.content

    state["messages"].append(("assistant", reply_text))
    return state

def guideline_update_node(state: State) -> State:
    """
    1. get current guidelines
    2. LLM proposes updated guidelines
    3. store them in `state["proposed_guidelines"]`
    4. show user the new guidelines
    """
    current = get_guideline()
    user_text = get_last_user_message(state["messages"]) or ""

    # Propose new guidelines
    updated = llm(
        guideline_update_prompt.format(
            user_input=user_text, 
            current_guidelines=current
        )
    ).strip()

    state["proposed_guidelines"] = updated
    # In a real flow, you might ask user to confirm. Here, let's just show them:
    reply = f"Here is the proposed new guideline:\n\n{updated}"
    state["messages"].append(("assistant", reply))

    # (Optional) auto-update or prompt user to confirm. We'll just auto-update for minimal code:
    update_guideline(updated)

    return state

##############################################################################
# 9. Build StateGraph
##############################################################################
from langgraph.graph import START, StateGraph

builder = StateGraph(State)

builder.add_node("classifier", classification_node)
builder.add_node("chitchat", chitchat_node)
builder.add_node("guideline_update", guideline_update_node)

# Start -> classification
builder.add_edge(START, "classifier")

# If classification == "chitchat" => go chitchat
# If classification == "guideline_update" => go guideline_update
def route_classification(state: State):
    if state.get("classification") == "guideline_update":
        return "guideline_update"
    else:
        return "chitchat"

builder.add_conditional_edges("classifier", route_classification)

# For simplicity, after chitchat => back to classification
builder.add_edge("chitchat", "classifier")
# after guideline_update => back to classification
builder.add_edge("guideline_update", "classifier")

memory = MemorySaver()
graph = builder.compile(memory)

##############################################################################
# 10. Usage Example
##############################################################################
if __name__ == "__main__":
    print("Minimal CS Chatbot. Type 'exit' to quit.\n")
    state: State = {
        "messages": [],
        "classification": "",
        "proposed_guidelines": ""
    }
    config = {"configurable": {"thread_id": "user_123"}}

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        state["messages"].append(("user", user_input))
        state = graph.invoke(state, config=config)

        # Print any new assistant messages
        for role, content in state["messages"]:
            if role == "assistant":
                print(f"Assistant: {content}")
        print()
