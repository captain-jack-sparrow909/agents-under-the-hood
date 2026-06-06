# in this file, we want to go ahead without using langchain
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
import ollama

MAX_ITERATIONS = 10
MODEL = "qwen3.6:35b-a3b"


# tools
@traceable(run_type='tool')
def get_product_price(product_name: str) -> float:
    """Get the price of a product"""
    prices = {
        "laptop": 1.00,
        "keyboard": 0.50,
        "mouse": 2.00,
    }
    return prices.get(product_name, 0.00)

@traceable(run_type='tool')
def get_product_discount(price: float, discount_tier: str) -> float:
    """Get the discount for a product"""
    discounts = {
        "silver": 5,
        "bronze": 12,
        "gold": 23,
    }
    return round(price * (1 - discounts.get(discount_tier, 0) / 100), 2)

# Difference 1:
# without @tool, we must manually define JSON schema for each function
# this is exactly what langchain @tool decorator do for us from the function type hints and docstring

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "look up the price of the product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product name eg: 'laptop', 'keyboard', mouse'"
                    }
                },
                "required": ["product_name"],
            }

        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_discount",
            "description": "Apply discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "the original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver' or 'gold'"
                    }
                },
                "required": ["price", "discount_tier"],
            }

        }
    }
]

# note: ollama can also generate these schemas if you pass the functions directly as tools, however that requires your docstring to follow
# google docstring format 



#---Helper: traced ollama call --
# difference 2: without langchain, we must manually trace LLM call for langsmith

@traceable(name="ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)



# agent loop

@traceable
def agent_loop(question: str):
    """run agent"""
    tools_dict = {"get_product_price": get_product_price, "get_product_discount": get_product_discount}

    messages = [
        # difference 3: format of messages changed, this is specific to ollama
        {
            "role": "system",
            "content":(
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
        },
        {
            "role": "user",
            "content": question
        },
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print("\n==iteration==", iteration)
        # difference 4: use ollama.chat() directly instead of llm invoking
        response = ollama_chat_traced(messages)
        ai_message = response.message

        tools_call = ai_message.tool_calls #this will decide whether the loop ends or tool call will happen

        #if no tools call then it's the final answer
        if not tools_call:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call from the tools_call returned - force one tool per iteration; as LLMs now can return multiple tool calls at once
        tool_call = tools_call[0]
        # difference 5: this is how to get the fucntion name and args
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"selected tool: {tool_name} with args : {tool_args}")
        
        #now we go ahead and execute this tool:
        tool_to_use = tools_dict.get(tool_name)  #tooL_to_use is going to be a langchain tool which we can invoke
        if tool_to_use is None:
            raise ValueError(f"tool {tool_name} not found")
        
        # difference 6: since we don't have .invoke, we call this function directly with the arguments
        observation = tool_to_use(**tool_args)
        print(f"tool result: {observation}")

        #now we want to make LLM remember what were the results from both the LLM and the tools
        messages.append(ai_message)
        messages.append({ "role" : "tool", "content": str(observation) })
        
    print("MAX iterations reached without a final answer")
    return None



if __name__ == "__main__":
    print("Hello Langchain, bind_tools\n")
    result = agent_loop("What is the price of laptop after applying gold discount?")