from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq = Groq()

def classify_with_llm(log_msg):
    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags. 
    Log message: {log_msg}'''
    chat_completion =groq.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ])
    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    print(classify_with_llm("Case escalation for ticket ID 12345 failed because the assigned suppoert agent is on leave."))
    print(classify_with_llm("system reboot intiated by user User789"))
    print(classify_with_llm("The 'Repoert Generator' module is deprecated and will be removed in future releases."))
