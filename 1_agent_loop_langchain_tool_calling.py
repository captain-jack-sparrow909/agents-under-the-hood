from unittest import result
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.6:35b-a3b"


# tools
@tool
def get_product_price(product_name: str) -> float:
    """Get the price of a product"""
    prices = {
        "laptop": 1.00,
        "keyboard": 0.50,
        "mouse": 2.00,
    }
    return prices.get(product_name, 0.00)

@tool
def get_product_discount(price: float, discount_tier: str) -> float:
    """Get the discount for a product"""
    discounts = {
        "silver": 5,
        "bronze": 12,
        "gold": 23,
    }
    return round(price * (1 - discounts.get(discount_tier, 0) / 100), 2)



# agent loop

@traceable
def agent_loop(question: str):
    """run agent"""
    tools = [get_product_price, get_product_discount]
    tools_dict = {t.name : t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools=tools)

    messages = [
        SystemMessage(
            content=(
                "You are a shopping assistant."
                "You have access to product catalog and a discount tool."
                "STRICT-RULES: you must follow these exactly:\n"
                "1. Never guess the price any product price."
                "You must call get_product_price tool to get the real price.\n"
                "2. Only apply discount using get_product_discount only after you've received the price from get_product_price"
                "pass the exact price returned by get_product_price - do not pass a madeup number.\n"
                "3. Never calculate discount using math, always use get_product_discount tool.\n"
                "4. If the user doesn't specify a discount tier,"
                "ask them which tier to use, don't assume one."
            ),
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print("\n==iteration==", iteration)
        ai_message = llm_with_tools.invoke(messages)

        tools_call = ai_message.tool_calls #this will decide whether the loop ends or tool call will happen

        #if no tools call then it's the final answer
        if not tools_call:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call from the tools_call returned - force one tool per iteration; as LLMs now can return multiple tool calls at once
        tool_call = tools_call[0]
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args')
        tools_id = tool_call.get('id')

        print(f"selected tool: {tool_name} with args : {tool_args}")
        
        #now we go ahead and execute this tool:
        tool_to_use = tools_dict.get(tool_name)  #tooL_to_use is going to be a langchain tool which we can invoke
        if tool_to_use is None:
            raise ValueError(f"tool {tool_name} not found")
        observation = tool_to_use.invoke(tool_args)
        print(f"tool result: {observation}")

        #now we want to make LLM remember what were the results from both the LLM and the tools
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tools_id))
        
    print("MAX iterations reached without a final answer")
    return None



if __name__ == "__main__":
    print("Hello Langchain, bind_tools\n")
    result = agent_loop("What is the price of laptop after applying gold discount?")