from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser


load_dotenv()

template=PromptTemplate(template="""generate detailed summary in 5-6 lines of following topic {topic}""",
                        input_variables=['topic'])

template2= PromptTemplate(template="""give me 3-5 five iportant keywords/points from the following summary : {summary}""",
                          input_variables=['summary'])

model= ChatAnthropic(model_name='claude-haiku-4-5')

parser= StrOutputParser()

chain= template | model | parser | template2 | model | parser
result= chain.invoke({'topic':'global warming'})

print(result)