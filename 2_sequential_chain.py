from llm_client import get_simple_client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

os.environ["LANGCHAIN_PROJECT"] = "2_sequential_chain"

report_generator = get_simple_client(
    input_variables=["topic"],
    template="You are an expert in {topic}. Please provide a detailed report of this topic.",
)

editor = get_simple_client(
    input_variables=["report"],
    template="You are a professional editor. Please edit the following report for clarity and conciseness: {report}",
    model="qwen/qwen3.8-27b"
)

summarizer = get_simple_client(
    input_variables=["edited_report"],
    template="You are a skilled summarizer. Please summarize the following report: {edited_report}"
)

config = {
    'tags': ["sequential_chain", "report_generation"],
    'metadata': {
        'models': ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
        'parser': "StrOutputParser"
                 }
}

chain = report_generator | editor | summarizer 

response = chain.invoke({"topic": "Time Travel"}, config=config)

print("Final Summary:")
print(response)