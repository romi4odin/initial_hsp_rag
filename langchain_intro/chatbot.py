import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (

    PromptTemplate,

    SystemMessagePromptTemplate,

    HumanMessagePromptTemplate,

    ChatPromptTemplate,

)
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.schema.runnable import RunnablePassthrough

from langchain.agents import(
    Tool,
    AgentExecutor,create_xml_agent
)

from langchain import hub

from langchain_intro.tools import get_current_time_wait_time
#A Tool is an interface that an agent uses to interact with a function.

"""reads and stores environment variables from the .env file. 
By default it understands that the .env file is present in the same directory ,
in case .env file is added in different directory then please add in the arguments"""

dotenv.load_dotenv() 

review_template_str = """Your job is to use patient

reviews to answer questions about their experience at

a hospital. Use the following context to answer questions.

Be as detailed as possible, but don't make up any information

that's not from the context. If you don't know an answer, say

you don't know.


{context}

"""
#prompt template specifically for SystemMessage
review_system_prompt = SystemMessagePromptTemplate(

    prompt=PromptTemplate(

        input_variables=["context"],

        template=review_template_str,

    )

)

#eview_human_prompt for the HumanMessage
review_human_prompt = HumanMessagePromptTemplate(

    prompt=PromptTemplate(

        input_variables=["question"],

        template="{question}",

    )

)

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(

    input_variables=["context", "question"],

    messages=messages,

)


"""instantiating a chatopenai model with gpt 3.5 turbo as base model and temperature
temperature 0 is for getting deterministic answer although it doesn't guarantee that"""

chat_model = ChatAnthropic(model='claude-2.1', temperature=0)

output_parser = StrOutputParser()

REVIEWS_CHROMA_PATH = "chroma_data/"

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=embedding_function
)
reviews_retriever  = reviews_vector_db.as_retriever(k=10)

#that can pass questions through review_prompt_template and chat_model in a single function call
review_chain = {"context": reviews_retriever, "question": RunnablePassthrough()} | review_prompt_template | chat_model | output_parser


tools = [
    Tool(
        name="Reviews",
        func=review_chain.invoke,
        description="""Useful when you need to answer questions
        about patient reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_time_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]

hospital_agent_prompt = hub.pull("hwchase17/xml-agent-convo")

hospital_agent = create_xml_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools
)

#agent to pass values to function

hospital_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
