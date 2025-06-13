prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
For list of answers, give in bulletin.

For the given medical values, analyse it and give a short report on it. Give medical sugessions and food diet.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""