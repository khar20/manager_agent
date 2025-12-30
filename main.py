import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Any

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- CONFIGURATION & LOGGING ---

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL = "gpt-5"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- DATABASE CONNECTION POOL ---

class DatabasePool:
    _pool = None

    @classmethod
    def initialize(cls):
        try:
            cls._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                dsn=DATABASE_URL
            )
            logger.info("Database connection pool initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    @classmethod
    def close(cls):
        if cls._pool:
            cls._pool.closeall()
            logger.info("Database connection pool closed.")

    @classmethod
    def get_conn(cls):
        if not cls._pool:
            raise Exception("Database pool not initialized")
        return cls._pool.getconn()

    @classmethod
    def put_conn(cls, conn):
        if cls._pool:
            cls._pool.putconn(conn)

def execute_sql_query_sync(query: str) -> str:
    conn = None
    try:
        conn = DatabasePool.get_conn()
        conn.autocommit = True 
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            if cur.description:
                result = cur.fetchall()
                return json.dumps(result, default=str)
            else:
                return json.dumps({"status": "success", "message": "Query executed successfully."})
    except Exception as e:
        logger.error(f"SQL Execution Error: {e}")
        return json.dumps({"status": "error", "error": str(e)})
    finally:
        if conn:
            DatabasePool.put_conn(conn)

# --- LIFESPAN MANAGER ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    DatabasePool.initialize()
    yield
    DatabasePool.close()

app = FastAPI(lifespan=lifespan)

# --- TOOLS DEFINITION ---

tools = [
    {
        "type": "web_search"
    },
    {
        "type": "function",
        "name": "run_database_query",
        "description": "Execute a generic SQL query to GET, ADD, or UPDATE data.",
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
]

# --- PROMPTS & SCHEMAS ---

DB_SCHEMA = """
user_role_type: 'Admin', 'Manager', 'Employee'
project_status_type: 'Planning', 'In Progress', 'Completed', 'Blocked', 'Cancelled'
task_priority_type: 'High', 'Medium', 'Low'
task_status_type: 'To Do', 'In Progress', 'Review', 'Done'
asset_status_type: 'In Stock', 'In Use', 'Maintenance', 'Missing', 'Retired'

users (id, email, full_name, user_role, is_active)
clients (id, name, contact_email, phone)
assets (id, asset_name, serial_number, location, purchase_value, asset_status)
projects (id, client_id, project_manager_id, project_name, description, budget_hours, start_date, due_date, project_status)
tasks (id, project_id, assigned_user_id, task_name, estimated_hours, priority, task_status)
asset_allocations (id, asset_id, task_id, quantity, allocated_from, allocated_until)
"""

def get_instructions():
    now = datetime.now()
    formatted_date = now.strftime("%A, %B %d, %Y at %I:%M %p")
    return f"""
    Role: You are an assistant agent with access to a Company Database and the Internet.
    Current date: {formatted_date}
    Database Schema: {DB_SCHEMA}
    Instructions:
    1. Use 'run_database_query' for internal data.
    2. Use 'web_search' for external data.
    3. Always return a final response to the user.
    """

# --- API ENDPOINTS ---

class AgentRequest(BaseModel):
    query: str = Field(alias="query") 
    session_id: Optional[str] = Field(default=None, alias="session_id") 

    class Config:
        populate_by_name = True

@app.post("/agent")
async def run_agent(request: AgentRequest):
    logger.info(f"Received agent request: {request.query}")
    
    conversation_input = [
        {"role": "user", "content": request.query}
    ]

    while True:
        # Async call to OpenAI
        response = await aclient.responses.create(
            model=MODEL,
            instructions=get_instructions(),
            input=conversation_input,
            tools=tools
        )

        function_calls = [i for i in response.output if i.type == "function_call"]

        if not function_calls:
            return {"response": response.output_text}

        # Update history with assistant's decision
        for item in response.output:
            if item.type == "message":
                conversation_input.append({"role": "assistant", "content": item.content})
            elif item.type == "function_call":
                conversation_input.append({
                    "type": "function_call",
                    "call_id": item.call_id,
                    "function": {
                        "name": item.function.name,
                        "arguments": item.function.arguments
                    }
                })

        # Execute tools
        for call in function_calls:
            if call.function.name == "run_database_query":
                args = json.loads(call.function.arguments)
                sql_query = args.get("query")
                
                logger.info(f"Executing SQL: {sql_query}")
                
                # Run blocking DB call in thread pool to prevent blocking event loop
                result = await run_in_threadpool(execute_sql_query_sync, sql_query)
                
                conversation_input.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result
                })