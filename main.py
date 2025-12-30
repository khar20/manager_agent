import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

# CONFIGURATION

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL = "gpt-4o"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set")

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# DATABASE LAYER

class DatabasePool:
    _pool = None

    @classmethod
    def initialize(cls):
        cls._pool = psycopg2.pool.ThreadedConnectionPool(1, 20, dsn=DATABASE_URL)
        logger.info("Database pool initialized")

    @classmethod
    def close(cls):
        if cls._pool:
            cls._pool.closeall()
            logger.info("Database pool closed")

    @classmethod
    def get_conn(cls):
        return cls._pool.getconn()

    @classmethod
    def put_conn(cls, conn):
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
            return json.dumps({"status": "success", "message": "Query executed successfully."})
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return json.dumps({"status": "error", "error": str(e)})
    finally:
        if conn:
            DatabasePool.put_conn(conn)

# APP LIFECYCLE

@asynccontextmanager
async def lifespan(app: FastAPI):
    DatabasePool.initialize()
    yield
    DatabasePool.close()

app = FastAPI(lifespan=lifespan)

# SCHEMA & PROMPTS

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

# API ENDPOINT

class AgentRequest(BaseModel):
    query: str = Field(alias="query")
    session_id: Optional[str] = Field(default=None, alias="session_id")

    class Config:
        populate_by_name = True

@app.post("/agent")
async def run_agent(request: AgentRequest):
    logger.info(f"Received query: {request.query}")
    
    conversation_input = [
        {"role": "user", "content": request.query}
    ]

    while True:
        response = await aclient.responses.create(
            model=MODEL,
            instructions=get_instructions(),
            input=conversation_input,
            tools=tools
        )

        function_calls = [i for i in response.output if i.type == "function_call"]

        if not function_calls:
            return {"response": response.output_text}

        for item in response.output:
            if item.type == "message":
                conversation_input.append({"role": "assistant", "content": item.content})
            elif item.type == "function_call":
                conversation_input.append({
                    "type": "function_call",
                    "call_id": item.call_id,
                    "function": {
                        "name": item.name, 
                        "arguments": item.arguments
                    }
                })

        for call in function_calls:
            if call.name == "run_database_query":
                args = json.loads(call.arguments)
                sql_query = args.get("query")
                
                logger.info(f"Executing SQL: {sql_query}")
                result = await run_in_threadpool(execute_sql_query_sync, sql_query)
                
                conversation_input.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result
                })