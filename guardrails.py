import re

MAX_QUESTION_LEN = 500
ROW_LIMIT = 200
QUERY_TIMEOUT_SEC = 30
WRITE_WORDS = (r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach"
               r"|pragma|vacuum|reindex|grant|revoke)\b")
REFUSAL = "Nice try deeapk bhaii, Sorry i should not answer for that"


class Blocked(Exception):
    """Raised when a question or a generated query fails a guardrail."""


def check_question(question: str) -> str:
    q = (question or "").strip()
    if not q:
        raise Blocked("Please ask a question about the data.")
    if len(q) > MAX_QUESTION_LEN:
        raise Blocked(f"Question too long (max {MAX_QUESTION_LEN} characters).")
    if re.search(WRITE_WORDS, q, re.I):  
        raise Blocked(REFUSAL)
    return q


def check_sql(sql: str, allowed_tables) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = sql.strip().rstrip(";").strip()

    if not sql:
        raise Blocked("The model did not return any SQL.")
    if ";" in sql:
        raise Blocked("Only a single statement is allowed.")
    if not re.match(r"(?is)^(select|with)\b", sql):
        raise Blocked("Only SELECT queries are allowed.")
    if re.search(WRITE_WORDS, sql, re.I):
        raise Blocked("Data-modifying SQL is not allowed.")

    scan = re.sub(r"'[^']*'", "''", sql)  
    used = {t.lower() for t in re.findall(r"(?is)\b(?:from|join)\s+[\"`\[]?(\w+)", scan)}
    ctes = {c.lower() for c in re.findall(r"(?is)\b(\w+)\s+as\s*\(", scan)}
    unknown = used - {t.lower() for t in allowed_tables} - ctes
    if unknown:
        raise Blocked(f"Query references unknown table(s): {', '.join(sorted(unknown))}")

    if not re.search(r"(?is)\blimit\s+\d+", sql):
        sql += f" LIMIT {ROW_LIMIT}"
    return sql
