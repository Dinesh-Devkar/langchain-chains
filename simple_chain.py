from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

template= PromptTemplate(template= """ explain in two lines about the {topic}""",
                         input_variables=['topic'])

model=ChatAnthropic(model_name='claude-haiku-4-5')

parser=StrOutputParser()

chain = template | model | parser

result= chain.invoke({'topic':'space'})

print(result)

print("="*30)

print(chain.get_graph().print_ascii())