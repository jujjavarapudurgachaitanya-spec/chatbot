"""Text-to-SQL: a LangGraph agent on Gemini that explores the database, writes SQL, and fixes its own errors.

    python chatbot.py                 # interactive CLI
    python chatbot.py "top 5 vendors by PO value"
"""

import operator
import os
import sqlite3
import sys
import time
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph

import guardrails as g

load_dotenv()

DB = Path(__file__).parent / "data" / "chatbot.db"
MODEL = "gemini-3.5-flash-lite"
MAX_STEPS = 12  # every model turn and every tool run counts as one step

SYSTEM = """You answer questions about a supply-chain SQLite warehouse.
Tables: {tables}
Call describe_tables for the one or two tables that look relevant, then run_sql with one
SELECT (or WITH) statement. Never write or modify data.
If run_sql returns an ERROR, fix the SQL and call it again. Once it returns rows,
reply with one short sentence about them and stop.
JOIN only when the key columns genuinely match. These tables use different product keys
(some have `material`, others a `12nc` code) - when they do not line up, answer from the
single most relevant table rather than refusing.
If the question is vague or subjective, pick a sensible interpretation and answer it.
For "top N <thing>" questions, GROUP BY that thing and return the measure beside it.
Values are stored as text, so CAST(col AS REAL) before aggregating numbers.
Double-quote any identifier starting with a digit ("12nc", "12m_value")."""


# ---- Tools the agent can call ----

def connect():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)  # read-only: writes are impossible


@lru_cache(maxsize=1)
def tables() -> list[str]:
    """Every table name, read from SQLite itself."""
    with closing(connect()) as conn:
        return [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    
def describe_tables(names: str) -> str:
    """Comma-separated table names -> the column names of each table."""
    with closing(connect()) as conn:
        return "\n".join(
            f"{t}: " + ", ".join(r[1] for r in conn.execute("SELECT * FROM pragma_table_info(?)", (t,)))
            for t in (n.strip() for n in names.split(",")) if t in tables())


_last: dict = {}


def run_sql(sql: str) -> str:
    """Run one read-only SQLite SELECT and return its first rows, or an ERROR to fix."""
    conn = connect()
    deadline = time.monotonic() + g.QUERY_TIMEOUT_SEC
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 100_000)
    try:
        checked = g.check_sql(sql, tables())  # guardrails gate every query
        df = pd.read_sql_query(checked, conn)
    except Exception as e:  # blocked, bad column, or timeout: the agent reads it and retries
        return f"ERROR: {e.__cause__ or e}"
    finally:
        conn.close()
    _last.update(sql=checked, df=df)
    return df.head(10).to_string(index=False) if len(df) else "0 rows"


TOOLS = {f.__name__: f for f in (describe_tables, run_sql)}


# ---- The LangGraph agent: model -> tools -> model ... until the model stops calling tools ----

class State(TypedDict):
    messages: Annotated[list[types.Content], operator.add]  # each node appends to the conversation


@lru_cache(maxsize=1)
def client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise g.Blocked("Set GEMINI_API_KEY as an env var or in a .env file next to this script.")
    return genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=60_000, retry_options=types.HttpRetryOptions(attempts=2)))


def tool_calls(state: State) -> list:
    return [p.function_call for p in state["messages"][-1].parts or [] if p.function_call]


def call_model(state: State) -> dict:
    """Gemini decides the next step: call a tool, or give the final answer."""
    reply = client().models.generate_content(model=MODEL, contents=state["messages"], config=types.GenerateContentConfig(
        system_instruction=SYSTEM.format(tables=", ".join(tables())),
        tools=list(TOOLS.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
    return {"messages": [reply.candidates[0].content]}


def call_tools(state: State) -> dict:
    """Run every tool Gemini asked for and hand the results back to it."""
    results = [types.Part.from_function_response(name=c.name, response={"result": TOOLS[c.name](**c.args)})
               for c in tool_calls(state)]
    return {"messages": [types.Content(role="user", parts=results)]}


@lru_cache(maxsize=1)
def agent():
    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", lambda s: "tools" if tool_calls(s) else END)
    graph.add_edge("tools", "model")
    return graph.compile()


def ask(question: str) -> tuple[str, pd.DataFrame]:
    q = g.check_question(question)  # dangerous questions stop here, before any API call
    client()  # fail fast with a clear message if the API key is missing
    _last.clear()
    try:
        agent().invoke({"messages": [types.Content(role="user", parts=[types.Part(text=q)])]},
                       {"recursion_limit": MAX_STEPS})
    except Exception as e:
        raise g.Blocked(f"Agent failed ({type(e).__name__}): {e}") from e
    if "df" not in _last:
        raise g.Blocked("The agent could not produce a working query.")
    return _last["sql"], _last["df"]


if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else iter(lambda: input("\n> "), "")
    for question in questions:
        try:
            sql, df = ask(question)
            print(f"\n{sql}\n\n{df.to_string(max_rows=20)}")
        except (g.Blocked, sqlite3.Error) as e:
            print(f"\n[blocked] {e}")
