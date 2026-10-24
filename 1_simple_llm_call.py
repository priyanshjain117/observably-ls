from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

prompt= PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question: {question}"
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=API_KEY
               )

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke({"question": "What is the capital of France?"})

print(response)