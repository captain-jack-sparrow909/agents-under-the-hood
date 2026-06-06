# in this file, we want to go ahead without using langchain and use raw ReAct prompt
from dotenv import load_dotenv
load_dotenv()

import re   #regular expression, needed to extraction data
import inspect  #to get meta-data on the functions we'll be using as tools
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
    price = float(price)
    return round(price * (1 - discounts.get(discount_tier, 0) / 100), 2)

# Difference 1:
# no longer have schema, defining tools like this
tools = {
    "get_product_price": get_product_price,
    "get_product_discount": get_product_discount
}

#but to let LLM know about our tools, we define another function for that:
def get_tool_description(tools_dict):
    description = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers 
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)  #to get function's meta data
        doc_string = inspect.getdoc(original_function) or ""
        description.append(f"{tool_name}{signature} - {doc_string}")

    return "\n".join(description)    #as this is going to be fed into LLM, we convert the list into a string

tool_description = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

# writing the prompt, along with all the rules:
react_prompt = f"""
    STRICT-RULES: you must follow these exactly:\n
                1. Never guess the price any product price.
                You must call get_product_price tool to get the real price.\n
                2. Only apply discount using get_product_discount only after you've received the price from get_product_price
                pass the exact price returned by get_product_price - do not pass a madeup number.\n
                3. Never calculate discount using math, always use get_product_discount tool.\n
                4. If the user doesn't specify a discount tier,
                ask them which tier to use, don't assume one.

    Answer the following questions as best you can. You have access to the following tools:
    {tool_description}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {{question}}
    Thought: """


#---Helper: traced ollama call --
# difference 2: without langchain, we must manually trace LLM call for langsmith

@traceable(name="ollama chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options) #earlier we passed tools, we're dependent on raw intelligence of the LLM to do that for us from the react prompt



# agent loop

@traceable(name="ollama agent loop")
def agent_loop(question: str):
    """run agent"""
    prompt = react_prompt.format(question=question)
    scratch_pad = ""


    for iteration in range(1, MAX_ITERATIONS+1):
        print("\n==iteration==", iteration)
        full_prompt = prompt + scratch_pad
        # difference 4: use ollama.chat() directly instead of llm invoking
        response = ollama_chat_traced(
                    model=MODEL, 
                    messages=[{"role": "user", "content": full_prompt}],
                    options={"stop": ["\nObservation"], "temperature": 0}
        )
        output = response.message.thinking

        print(f"LLM output:\n {output}")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"\nFinal Answer: {final_answer}")

        #parse tool calls from the raw text with regex - fragile if LLM doesn't follow the format
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        if not action_match or not action_input_match:
            print("coudn't parse Action or Action input from the LLM output")
            break
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()


        print(f"selected tool: {tool_name} with args : {tool_input_raw}")

        #the arguments still in raw form and need to be handled:
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"Tool executing: {tool_name}({args})...")

        if not tool_name in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))
        
        scratch_pad += f"{output}\nObservation: {observation}\nThought:"
        
    print("MAX iterations reached without a final answer")
    return None



if __name__ == "__main__":
    print("Hello Langchain, bind_tools\n")
    result = agent_loop("What is the price of laptop after applying gold discount?")