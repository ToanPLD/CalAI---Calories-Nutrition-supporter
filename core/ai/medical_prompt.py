SYSTEM_PROMPT = """
You are a certified nutrition doctor.

Provide:
- clear explanation
- health impact
- recommendation

Be professional like a doctor.
"""


def build_prompt(query, df):

    return f"""
{SYSTEM_PROMPT}

User question:
{query}

Data:
{df.to_string()}

Answer:
"""