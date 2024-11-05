
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate

load_dotenv()

memory = MemorySaver()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


### Construct retriever ###
confluence_loader = ConfluenceLoader(
    url="https://docs-chatbot.atlassian.net/",
    username="kaszuraszymon@gmail.com",
    api_key=os.getenv("CONFLUENCE_API_KEY"), 
    space_key="~603690b2ac6e4e0069e77ed4", 
    include_attachments=False, 
    limit=50
)

docs = confluence_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

### Build retriever tool ###
tool = create_retriever_tool(
    retriever,
    "documentation_retriever",
    "Searches and returns excerpts from the Confluence documentation. Include any source URLs provided in the retrieved content in your final response. If there's no information in the Confluence about the topic, it responds that it doesn't know.",
    document_prompt=document_prompt
)
tools = [tool]

agent_executor = create_react_agent(llm, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

system_message = SystemMessage(content= """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the question is to request links, please only return the source links with no answer.
2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following links** and add the source links as a list.
""")
query = "how to restart ec2?"
for event in agent_executor.stream(
    {"messages": [system_message, HumanMessage(content=query)]},
    config=config,
    stream_mode="values"
):
    event["messages"][-1].pretty_print()
