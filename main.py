import asyncio
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


async def call_model(state: MessagesState):
    chain = prompt | model
    response = await model.ainvoke(state["messages"])
    return {'messages': [*state["messages"], response]}


def create_workflow():
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


async def main():
    app = create_workflow()
    config = {"configurable": {"thread_id": "abc123"}}

    query = "Hi, I'm Bob"
    input_messages = [HumanMessage(content=query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
