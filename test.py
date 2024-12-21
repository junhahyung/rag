import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

openai_api_key = os.environ['OPENAI_API_KEY']

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage("Translate the following from English to Korean"),
    HumanMessage("hi!"),
]

result = model.invoke(messages)
print(result)