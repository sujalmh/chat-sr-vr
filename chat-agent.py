import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Dict, Any
import requests

load_dotenv()
SQL_API_ENDPOINT = "http://35.200.156.175:8074/query"
VECTOR_API_ENDPOINT = "http://35.200.156.175:8787/search-topN"

def call_sql_api(question: str) -> Dict[str, Any]:
    """
    Calls the external SQL API with a natural language question
    and returns the JSON response.
    """
    print(f"\nCalling SQL API at {SQL_API_ENDPOINT} with question: '{question}'")
    headers = {"Content-Type": "application/json", "access_token": "api-12345"}
    
    params = {"user_query": question}
    try:
        
        response = requests.get(SQL_API_ENDPOINT, headers=headers, params=params, timeout=300) 
        response.raise_for_status() 
        api_response = response.json()
        print(f"SQL API Response: {json.dumps(api_response, indent=2)}")
        return {"status": "success", "question": question, "data": api_response["result"]}
    except requests.exceptions.RequestException as e:
        print(f"Error calling SQL API: {e}")
        return {
            "status": "error",
            "message": f"Failed to connect to SQL API: {e}",
            "data": None
        }
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from SQL API. Response text: {response.text}")
        return {
            "status": "error",
            "message": "Failed to decode JSON response from SQL API.",
            "data": response.text 
        }
    except:
        print(f"Unexpected error occurred: {response.text}")
        return {
            "status": "error",
            "message": "An unexpected error occurred.",
            "data": response.text 
        }

def call_vector_search_api(question: str) -> Dict[str, Any]:
    """
    Calls the external Vector Search API with a question/keywords
    and returns the JSON response.
    """
    print(f"\nCalling Vector Search API at {VECTOR_API_ENDPOINT} with question: '{question}'")
    headers = {"Content-Type": "application/json", "access_token": "api-12345"}
    
    payload = json.dumps({"question": question})
    try:
        response = requests.post(VECTOR_API_ENDPOINT, headers=headers, data=payload, timeout=60) 
        response.raise_for_status() 
        api_response = response.json()
        print(f"Vector Search API Response: {json.dumps(api_response, indent=2)}")
        results = []
        for result in api_response["retrieved_results"]:
            
            snippet = {
                "content": result.get("content", "No Title"),
                "distance": result.get("distance", "No Snippet"),
                "source": result.get("source", "No URL"),
                "page": result.get("page", "No Page Number"),
                "reference": result.get("reference", "No Reference"),
                "cross_score": result.get("cross_score", "No Cross Score"),
                "date": result.get("date", "No Date")
            }
            results.append({"content": snippet["content"], "source": snippet["source"], "date": snippet["date"]})
        return {"status": "success", "question": question, "data": results}
    except requests.exceptions.RequestException as e:
        print(f"Error calling Vector Search API: {e}")
        return {
            "status": "error",
            "message": f"Failed to connect to Vector Search API: {e}",
            "data": None
        }
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from Vector Search API. Response text: {response.text}")
        return {
            "status": "error",
            "message": "Failed to decode JSON response from Vector Search API.",
            "data": response.text 
        }




class ExternalSqlApiTool(BaseTool):
    name: str = "external_sql_api"
    description: str = (
        "Useful for retrieving structured Indian economic data via an external API. "
        "Input should be a concise question in natural language that can be directly "
        "passed to the SQL API (e.g., 'What is India's GDP in 2023?', 'Show unemployment rate for last 5 years'). "
        "Returns a JSON object containing the API response, including data if successful."
    )

    def _run(self, question: str) -> str:
        """Use the tool to call the external SQL API."""
        response = call_sql_api(question)
        
        return json.dumps(response)

    
    
    
    
    


class ExternalVectorSearchApiTool(BaseTool):
    name: str = "vector_search_api_tool"
    description: str = (
        "Useful for retrieving information from unstructured Indian economic documents "
        "via an external semantic search API. "
        "Input should be a concise question or keywords in natural language that can be directly "
        "passed to the Vector Search API (e.g., 'recent monetary policy changes', 'PLI scheme manufacturing'). "
        "Returns a JSON object containing the API response, including document snippets if successful."
    )

    def _run(self, question: str) -> str:
        """Use the tool to call the external Vector Search API."""
        response = call_vector_search_api(question)
        
        return json.dumps(response)

    
llm = ChatOllama(model="mistral:instruct", extract_reasoning=True)


sql_api_tool = ExternalSqlApiTool()
vector_search_api_tool = ExternalVectorSearchApiTool()

tools = [sql_api_tool, vector_search_api_tool]




prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant specializing in Indian economic data. Your goal is to answer user questions strictly by using the provided external API tools — the SQL API and the Vector Search API — with no reliance on your pre-trained knowledge or any context outside of the API responses.

    You have access to two tools that call external APIs:
    1. `external_sql_api`: Use this for questions requiring structured data, like specific economic indicators, historical data series, or figures stored in a database.
    2. `vector_search_api_tool`: Use this for questions requiring information from unstructured documents, such as policy details, analyses, reports, or general economic concepts.

**Detailed Strategy:**
    1. **Analyze Query:** Break down the user's question to determine if the answer requires structured data (SQL) or unstructured text data (Vector Search).
    2. **Execute API Calls:**
        - Only use the SQL API for data that is in structured form, such as numerical data (GDP, inflation, etc.), or to support text data with numerical data frmo Vector Search API.
        - Use the Vector Search API for unstructured text data (policies, reports, etc.), also to support data from SQL API. Ensure the search terms are focused on retrieving exact documents or snippets relevant to the query.
    3. **Do Not Use Pre-trained Knowledge:** You should not use any information from your training or pre-existing knowledge base. All your answers should be based solely on the data retrieved from the SQL and Vector Search APIs.
    4. **Dual API Calls:**
        - Structured Data First: Always start by querying `external_sql_api` for relevant numerical data.
        - Textual Context: Then query `vector_search_api_tool` using related keywords or timeframes derived from the SQL results to fetch explanatory text or policy context.
        - If the initial thought identifies textual context as critical, you may reverse the order, but ensure both tools are called.
    5. **Synthesize and Present Data:**
        - Combine the SQL results (e.g., tables, numbers) and Vector Search results (e.g., text snippets, documents).
        - If both data sources are used, differentiate them clearly in your response.
        - Cite which tool (SQL or Vector Search) provided which data.
    6. **Iterative Sub-query Generation:**
        - If the initial query is too broad or complex, break it down into smaller sub-queries.
        - Use the SQL API to get specific data points and the Vector Search API to retrieve relevant documents or reports.
        - If a question fails to yield results, break the query into smaller, more specific sub-queries.
    7. **Reasoning and Citations:**
        - Explicitly state why a specific tool was used, based on the query's needs.
        - Provide citations for the data used. For example, say, "As retrieved from the SQL database" or "According to the Vector Search API response."

    Always be clear about where your information comes from, and do not include any content not directly retrieved from the APIs.
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



def ask_indian_economy_agent(question: str):
    """
    Sends a question to the Indian Economy Agent which uses external APIs.
    """
    print(f"\nUser Query: {question}")

    try:
        
        
        
        result = agent_executor.invoke({"input": f"""User Question: {question}
        Required Analysis:
        1. Extract precise numerical data
        2. Find relevant policy documents/explanations
        3. Show how context explains numbers
"""})
        final_response = result['output']
        print(f"\nAgent's Final Response:\n{final_response}")
        return final_response

    except Exception as e:
        print(f"\nAn error occurred during agent execution: {e}")
        return "Sorry, I encountered an error while processing your request."


if __name__ == "__main__":
    print("Indian Economy Agent (API Mode) activated. Ask me about Indian economic data.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        
        ask_indian_economy_agent(user_input)


