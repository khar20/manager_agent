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
MODEL = "gpt-4o"

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Postgres AI Agent")

# --- 1. DATABASE UTILS ---

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

# --- 2. AGENT TOOLS DEFINITION ---

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_database_query",
            "description": "Execute a generic SQL query to GET, ADD, or UPDATE data based on the user request.",
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

# --- 3. SYSTEM PROMPT & SCHEMA ---

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
    Role: You are an assistant agent. Your goal is to facilitate the management tasks of the users. 
    Don't ask me for input before executing a task, execute the task via SQL and show me the result for validation.
    
    Current date: {formatted_date}
    
    Database Schema:
    {DB_SCHEMA}
    
    Instructions:
    1. You have ONE tool: 'run_database_query'. Use it for EVERYTHING.
    2. Convert the user's natural language request into a specific PostgreSQL query.
    3. Always select relevant columns to show the user what happened.
    4. If adding data, handle foreign keys appropriately.
    5. Be concise in your final response.
    """

# --- 4. API ENDPOINT (UPDATED FOR GO COMPATIBILITY) ---

class AgentRequest(BaseModel):
    query: str = Field(alias="query") 
    
    session_id: str | None = Field(default=None, alias="session_id") 

    class Config:
        populate_by_name = True

@app.post("/agent")
async def run_agent(request: AgentRequest):
    # Log the incoming query for debugging (visible in Easypanel logs)
    print(f"Received Query: {request.query} | Session: {request.session_id}")

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": request.query} # Use 'query' from the request
    ]

    # First call to LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    messages.append(response_message)

    # Check if the agent wants to run a SQL query
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "run_database_query":
                args = json.loads(tool_call.function.arguments)
                sql_query = args.get("query")
                
                print(f"Executing SQL: {sql_query}")
                tool_output = execute_sql_query(sql_query)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "run_database_query",
                    "content": tool_output
                })

        # Second call to LLM
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return {"response": final_response.choices[0].message.content}
    
    return {"response": response_message.content}