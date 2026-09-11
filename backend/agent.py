import os
import json
from dotenv import load_dotenv
from groq import Groq
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.tools import (
    search_documents,
    extract_financial_data,
    get_market_data,
    calculate_financial_metric
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search uploaded financial documents to answer questions about their content. Use this when the user asks anything about the uploaded PDF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_financial_data",
            "description": "Extract structured financial figures and metrics from the document as a clean table. Use when user wants specific numbers, ratios or financial data extracted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The financial topic to extract data about"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_data",
            "description": "Fetch live stock market data for any company. Use when user asks about current stock price, market cap, or live market information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol like AAPL for Apple"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_financial_metric",
            "description": "Calculate financial metrics like EMI, compound interest, or CAGR. Use when user wants to calculate something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_type": {
                        "type": "string",
                        "description": "Type of calculation: emi, compound_interest, or cagr"
                    },
                    "principal": {
                        "type": "number",
                        "description": "Principal amount"
                    },
                    "rate": {
                        "type": "number",
                        "description": "Interest rate percentage"
                    },
                    "years": {
                        "type": "number",
                        "description": "Number of years"
                    }
                },
                "required": ["metric_type"]
            }
        }
    }
]

# Arguments required for each calculate_financial_metric type.
# Used to give a clear message instead of a raw KeyError when the LLM
# forgets to extract one of these from the user's question.
REQUIRED_CALC_ARGS = {
    "emi": ["principal", "rate", "years"],
    "compound_interest": ["principal", "rate", "years"],
    "cagr": ["initial", "final", "years"]
}

def execute_tool(tool_name, tool_args):
    if tool_name == "search_documents":
        if "query" not in tool_args:
            return {"error": "No search query was provided."}
        return search_documents(tool_args["query"])

    elif tool_name == "extract_financial_data":
        if "topic" not in tool_args:
            return {"error": "No topic was provided to extract data about."}
        return extract_financial_data(tool_args["topic"])

    elif tool_name == "get_market_data":
        if "ticker" not in tool_args:
            return {"error": "No stock ticker was provided."}
        return get_market_data(tool_args["ticker"])

    elif tool_name == "calculate_financial_metric":
        metric_type = tool_args.get("metric_type")
        required = REQUIRED_CALC_ARGS.get(metric_type)

        if required is None:
            return {"error": f"Unknown metric type: {metric_type}"}

        missing = [arg for arg in required if arg not in tool_args]
        if missing:
            return {
                "error": f"Missing required value(s) for {metric_type}: {', '.join(missing)}. "
                         f"Please provide {', '.join(required)}."
            }

        return calculate_financial_metric(
            metric_type,
            **{k: v for k, v in tool_args.items() if k != "metric_type"}
        )

    else:
        return {"error": f"Unknown tool: {tool_name}"}


def call_groq(messages, error_context=""):
    """
    Wraps a Groq chat completion call with error handling.
    Returns the response text, or raises a RuntimeError with a clear
    message if the call fails or comes back empty.
    """
    try:
        GROQ_MODEL = "openai/gpt-oss-120b"
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages
        )
    except Exception as e:
        raise RuntimeError(f"AI service call failed{(' during ' + error_context) if error_context else ''}: {str(e)}")

    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError(f"AI service returned an empty response{(' during ' + error_context) if error_context else ''}.")

    return response.choices[0].message.content.strip()


def run_agent(user_question, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    reasoning_steps = []

    # Step 1 - Ask Groq which tool to use
    routing_prompt = f"""You are a financial document analyst. Based on the user question, decide which tool to use.

Available tools:
1. search_documents - Use for questions about document content
2. extract_financial_data - Use for extracting financial figures/tables
3. get_market_data - Use for live stock prices (needs ticker like AAPL)
4. calculate_financial_metric - Use for EMI, compound interest, CAGR calculations

User question: {user_question}

Reply with ONLY the tool name and arguments in this exact format:
TOOL: tool_name
ARGS: {{"key": "value"}}

Important argument names:
- search_documents needs: {{"query": "your search text"}}
- extract_financial_data needs: {{"topic": "what to extract"}}
- get_market_data needs: {{"ticker": "SYMBOL"}}
- calculate_financial_metric needs: {{"metric_type": "emi/compound_interest/cagr", "principal": number, "rate": number, "years": number}}
If no tool needed, reply:
TOOL: none
ARGS: {{}}"""

    try:
        routing_text = call_groq(
            [{"role": "user", "content": routing_prompt}],
            error_context="tool routing"
        )
    except RuntimeError as e:
        # Routing failed entirely (Groq down, timeout, etc). Fail gracefully
        # instead of crashing — tell the user plainly rather than a raw 500.
        return {
            "answer": f"Sorry, I couldn't process your question right now. {str(e)}",
            "reasoning_steps": [],
            "conversation_history": conversation_history
        }

    # Step 2 - Parse tool decision
    tool_name = "none"
    tool_args = {}
    routing_parse_failed = False

    for line in routing_text.split("\n"):
        if line.startswith("TOOL:"):
            tool_name = line.replace("TOOL:", "").strip()
        elif line.startswith("ARGS:"):
            try:
                tool_args = json.loads(line.replace("ARGS:", "").strip())
            except (json.JSONDecodeError, ValueError):
                tool_args = {}
                routing_parse_failed = True

    # Step 3 - Execute tool
    tool_result = None
    if tool_name != "none":
        reasoning_steps.append({
            "tool_called": tool_name,
            "arguments": tool_args
        })
        try:
            tool_result = execute_tool(tool_name, tool_args)
        except Exception as e:
            tool_result = {"error": str(e)}
    elif routing_parse_failed:
        # The model picked a tool-less response, but only because we
        # couldn't parse its ARGS line — flag this in reasoning so it's
        # visible in "View Agent Reasoning" rather than silently invisible.
        reasoning_steps.append({
            "tool_called": "none (routing response could not be parsed)",
            "arguments": {}
        })

    # Step 4 - Generate final answer
    if tool_result:
        final_prompt = f"""You are a financial document analyst assistant.

User question: {user_question}

Tool used: {tool_name}
Tool result: {json.dumps(tool_result)}

Based on this information, provide a clear and helpful answer to the user."""
    else:
        final_prompt = f"""You are a financial document analyst assistant.
Answer this question: {user_question}"""

    try:
        final_answer = call_groq(
            [{"role": "user", "content": final_prompt}],
            error_context="generating the final answer"
        )
    except RuntimeError as e:
        final_answer = f"Sorry, I found the information but couldn't generate a full answer. {str(e)}"

    conversation_history.append({"role": "user", "content": user_question})
    conversation_history.append({"role": "assistant", "content": final_answer})

    return {
        "answer": final_answer,
        "reasoning_steps": reasoning_steps,
        "conversation_history": conversation_history
    }


if __name__ == "__main__":
    print("=== Test 1: Document Question ===")
    result = run_agent("What is the total revenue of Tata Motors?")
    print("Answer:", result["answer"])
    print("Tools used:", result["reasoning_steps"])

    print("\n=== Test 2: Calculation ===")
    result = run_agent("Calculate EMI for principal 500000, rate 10.5%, 15 years")
    print("Answer:", result["answer"])
    print("Tools used:", result["reasoning_steps"])

    print("\n=== Test 3: Market Data ===")
    result = run_agent("What is the current stock price of Apple?")
    print("Answer:", result["answer"])
    print("Tools used:", result["reasoning_steps"])