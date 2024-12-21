from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import os
import yaml


class RAGChatbot:
    def __init__(self, txt_file_path):
        # Initialize the language model
        self.llm = ChatOpenAI(model_name="gpt-4o")
        
        # Load and process the text file
        with open(txt_file_path, 'r') as file:
            raw_text = file.read()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_texts(texts, embeddings)
        
        # Create retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )
        
        # Initialize chat history
        self.chat_history = []
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get a response based on the knowledge base
        """
        result = self.qa_chain({
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append((question, result["answer"]))
        
        return result["answer"]

# Example usage:
# chatbot = RAGChatbot(api_key="your-api-key", txt_file_path="path/to/your/file.txt")
# response = chatbot.ask("What is this document about?")
# print(response)

chatbot = RAGChatbot('data/data.txt')
# Interactive chat loop
print("Welcome to the chatbot! Type 'quit' to exit.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye!")
        break
        
    try:
        response = chatbot.ask(user_input)
        print("\nBot:", response)
    except Exception as e:
        print("\nError occurred:", str(e))
