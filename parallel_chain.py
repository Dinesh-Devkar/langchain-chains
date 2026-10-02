from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()


gpt_model=ChatOpenAI(model='gpt-4o')
claude_model=ChatAnthropic(model_name='claude-haiku-4-5')

prompt_1= PromptTemplate(template="""give me detailed 10-15 lines story of following topic {topic}.""",
                         input_variables=['topic'])

prompt_2= PromptTemplate(template="""give me notes of following story text : {story}""",
                         input_variables=['story'])

prompt_3= PromptTemplate(template="""generate quize on following story text : {story}""",
                         input_variables=['story'])

prompt_4=PromptTemplate(template="""merge following notes and quize of the same story into single output . \n notes : {notes} . \n quize : {quize}""",
                        input_variables=['notes','quize'])

parser= StrOutputParser()

report_chain = prompt_1 | gpt_model | parser

parallel_chain = RunnableParallel({
    'notes': prompt_2 | gpt_model | parser,
    'quize':prompt_3 | claude_model | parser
})

merge_chain = prompt_4 | gpt_model | parser

final_chain = report_chain | parallel_chain | merge_chain

result= final_chain.invoke({'topic':'Ramayana'})

print(result)

final_chain.get_graph().print_ascii()