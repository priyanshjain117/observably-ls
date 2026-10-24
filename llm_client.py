from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

def get_simple_client(input_variables=["question"], 
                      template="You are a helpful assistant. Answer the following question: {question}",
                      model="openai/gpt-oss-120b"):
    prompt= PromptTemplate(
        input_variables=input_variables,
        template=template
    )
    llm = ChatGroq(
        model=model,
        api_key=API_KEY
    )
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    return chain

