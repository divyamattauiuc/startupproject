opeanaiAPI = "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"

from matplotlib.pyplot import text
import openai

openai.api_key = opeanaiAPI

x = ""
while (x != "stop"):
    x = input("Enter a question")
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=x,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    )
    a = response["choices"]
    b = a[0]
    c = b["text"]
    print(c)
