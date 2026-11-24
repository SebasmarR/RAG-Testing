import transformers
from langchain.agents import create_agent
from langchain.tools import tool
from RAG import *
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

rag_preloading()

system_prompt = """
    You are a helpful assistant that can answer questions based on the context
    you've been trained. If you don't know the answer, just say that you don't know. Do not try to make up an answer.
    You also have access to a set of tools that can help you answer questions based on the context. Use these tools wisely and only when necessary.
    These tools will let you search for information in a set of documents that you can access with RAG. You can use these tools to find relevant information to answer questions. 
    If you don't have enough information to answer a question, you can use the tools to search for more information.
    Answer the questions as truthfully as possible based on the context you have. If you don't know the answer, just say that you don't know. Do not try to make up an answer.
    Answer in spanish
    """


@tool
def query_rag_folder(query: str) -> str:
    """
    Search for information in the documents folder using RAG.

    Args:
        query (str): The query to search for.

    Returns:
        str: The answer to the query based on the context.
    """
    print(f"Searching for: {query}")
    return query_search(query)


agent = create_agent(model="ollama:qwen3.5:9b",
                     system_prompt=system_prompt,
                     tools=[query_rag_folder],
                     checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": str(uuid7())}}


execution = True
while execution:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    response = agent.invoke(
        {"messages": [
            {"role": "user", "content": user_input}]},
        config=config,
    )

    if isinstance(response, dict) and "messages" in response:
        last_message = response["messages"][-1]
        # Como es un objeto AIMessage, extraemos directamente su .content
        if hasattr(last_message, "content"):
            print(f"Agent: {last_message.content}")
        else:
            print(f"Agent: {last_message}")
    else:
        print(f"Agent: {response}")
