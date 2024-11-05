import asyncio
from typing import Sequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


async def call_model(state: State):
    chain = prompt | model
    response = await chain.ainvoke(state)
    return {'messages': response}


def create_workflow():
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


async def main():
    app = create_workflow()
    config = {"configurable": {"thread_id": "abc123"}}
    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
        HumanMessage(content="what's my name?")
    ]
    trimmed_messages = await trimmer.ainvoke(messages)
    async for chunk, metadata in app.astream(
        {"messages": trimmed_messages, "language": "polski"},
        config,
        stream_mode="messages"
    ):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="|")


if __name__ == "__main__":
    asyncio.run(main())
