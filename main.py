import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL = "gpt-5"

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Project Management Agent")

# DATABASE UTILS

def execute_sql_query(query: str):
    """
    Executes a raw SQL query against the database.
    """
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            if cur.description:
                result = cur.fetchall()
                return json.dumps(result, default=str)
            else:
                return json.dumps({"status": "success", "message": "Query executed successfully."})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
    finally:
        if conn:
            conn.close()

# AGENT TOOLS DEFINITION

tools = [
    {
        "type": "web_search"
    },
    {
        "type": "function",
        "function": {
            "name": "run_database_query",
            "description": "Execute a generic SQL query to GET, ADD, or UPDATE data in the local Postgres database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The valid PostgreSQL query to execute."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# SYSTEM PROMPT & SCHEMA

DB_SCHEMA = """
-- ENUMS
user_role_type: 'Admin', 'Manager', 'Employee'
project_status_type: 'Planning', 'In Progress', 'Completed', 'Blocked', 'Cancelled'
task_priority_type: 'High', 'Medium', 'Low'
task_status_type: 'To Do', 'In Progress', 'Review', 'Done'
asset_status_type: 'In Stock', 'In Use', 'Maintenance', 'Missing', 'Retired'

-- TABLES
users (id, email, full_name, user_role, is_active)
clients (id, name, contact_email, phone)
assets (id, asset_name, serial_number, location, purchase_value, asset_status)
projects (id, client_id, project_manager_id, project_name, description, budget_hours, start_date, due_date, project_status)
tasks (id, project_id, assigned_user_id, task_name, estimated_hours, priority, task_status)
asset_allocations (id, asset_id, task_id, quantity, allocated_from, allocated_until)
"""

def get_system_prompt():
    now = datetime.now()
    formatted_date = now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    return f"""
    Role: You are an assistant agent with access to the project management database and the Internet.
    
    Current date: {formatted_date}
    
    Database Schema:
    {DB_SCHEMA}
    
    Instructions:
    1. Use 'run_database_query' for internal data (Projects, Clients, Tasks).
    2. Use 'web_search' (built-in) to find outside information (Market rates, Tech news, Competitors).
    3. You can combine them (e.g., "Find the exchange rate for USD to EUR and update the project budget").
    4. Always show the user the result of your actions.
    """

# API ENDPOINT

class AgentRequest(BaseModel):
    query: str = Field(alias="query") 
    session_id: str | None = Field(default=None, alias="session_id") 

    class Config:
        populate_by_name = True

@app.post("/agent")
async def run_agent(request: AgentRequest):
    print(f"Received Query: {request.query}")

    # conversation with system prompt + user query
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": request.query}
    ]

    response = client.responses.create(
        model=MODEL,
        input=messages,
        tools=tools
    )

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call.function.name == "run_database_query":
                # Execute the SQL
                args = json.loads(tool_call.function.arguments)
                sql_query = args.get("query")
                print(f"Executing SQL: {sql_query}")
                
                tool_output = execute_sql_query(sql_query)
                
                # Append result to history
                messages.append({
                    "role": "assistant",
                    "tool_calls": [tool_call] # Pass the original tool call object
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output
                })

        # Generate final answer
        final_response = client.responses.create(
            model=MODEL,
            input=messages,
            tools=tools
        )
        return {"response": final_response.output_text}

    return {"response": response.output_text}