from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
import yaml


class GuidelineManager:
    def __init__(self):
        # Initialize the language model
        self.llm = ChatOpenAI(model_name="gpt-4o")
        
    def summarize_and_save(self, user_request: str) -> str:
        """
        Summarize user request and save as guidelines
        """
        with open('data/data.txt', 'r') as file:
            data = file.read()
            
        # Create prompt for summarization
        # prompt = f"""
        # You are a helpful assistant that generates a clear guidelines for customer service employees.

        # You are talking to a manager of a company that has a customer service team.

        # Decide whether the new talk should be added to the guidelines.

        # Manager request: {user_request}

        # If yes, only say [yes].
        
        # If no, replay relevant answer to the manager.
        # """
        prompt = f"""
        당신은 고객 서비스 직원들을 위한 명확한 가이드라인을 생성하는 도움이 되는 비서야.
        
        너의 이름은 아리야.

        너 고객 서비스 팀을 보유한 회사의 관리자와 대화하고 있어.

        새로운 는대화가 가이드라인에 추가되어야 하는지 결정해봐.

        관리자 요청: {user_request}

        만약 그렇다면, [yes] 라고만 답해.
        
        만약 아니라면, 관리자에게 일상적으로 답하고, 해당 내용이 가이드라인에 추가될 필요 없다는 말은 하지마. 마지막에 저에게 주실 가이드라인이 있으면 말해달라고 안내해.
        """

        # Get response from GPT-4
        messages = [HumanMessage(content=prompt)]
        response = self.llm.generate([messages])
        response_text = response.generations[0][0].text


        if response_text != "[yes]":
            return response_text

        else:

            # prompt = f"""
            # You are a helpful assistant that generates a clear guidelines for customer service employees.

            # You are talking to a manager of a company that has a customer service team.

            # This is the current guidelines:
            # {data}

            # The manager is asking you to add a new guideline to the list.
            
            # Manager request: {user_request}
            
            # Provide a clear, concise summary in bullet points that can serve as guidelines.
            # """
            prompt = f"""
            당신은 고객 서비스 직원들을 위한 명확한 가이드라인을 생성하는 도움이 되는 비서입니다.

            당신은 고객 서비스 팀을 보유한 회사의 관리자와 대화하고 있습니다.

            이것이 현재 가이드라인입니다:
            {data}

            관리자가 목록에 새로운 가이드라인을 추가하도록 요청하고 있습니다.
            
            관리자 요청: {user_request}
            
            가이드라인으로 사용될 수 있는 명확하고 간결한 요약을 글머리 기호로 제공하십시오.
            """
            
            # Get response from GPT-4
            messages = [HumanMessage(content=prompt)]
            response = self.llm.generate([messages])
            summary = response.generations[0][0].text
            
            # Save to data.txt
            with open('data/data.txt', 'a') as file:
                file.write("\n\n--- New Guidelines ---\n")
                file.write(f"Original Request: {user_request}\n")
                file.write("Summary:\n")
                file.write(summary)

            # prompt = f"""
            # You are a helpful assistant that generates a clear guidelines for customer service employees.

            # You are talking to a manager of a company that has a customer service team.

            # The manager is asked you this: 
            
            # Manager request: {user_request}

            # And you have alreday edited the guidelines. 

            
            # This is the new guideline:
            # {summary}

            # And this is the original guidelines before the update:
            # {data}

            # Explain what you have added to the guidelines, and respond to the manager.
            # """

            prompt = f"""
            당신은 고객 서비스 직원들을 위한 명확한 가이드라인을 생성하는 도움이 되는 비서입니다.

            당신은 고객 서비스 팀을 보유한 회사의 관리자와 대화하고 있습니다.

            관리자가 다음과 같이 요청했습니다:
            
            관리자 요청: {user_request}

            그리고 당신은 이미 가이드라인을 수정했습니다.
            
            이것이 새로운 가이드라인입니다:
            {summary}

            그리고 이것이 업데이트 전의 원래 가이드라인입니다:
            {data}

            가이드라인에 추가한 내용을 설명하고 관리자에게 응답하십시오.
            """
            # Get response from GPT-4
            messages = [HumanMessage(content=prompt)]
            response = self.llm.generate([messages])
            response_text = response.generations[0][0].text

            return response_text

                

manager = GuidelineManager()

print("Welcome to the Guideline Manager! Type 'quit' to exit.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye!")
        break
        
    try:
        response = manager.summarize_and_save(user_input)
        print("\nBot:", response)
    except Exception as e:
        print("\nError occurred:", str(e))
