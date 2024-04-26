import json

from openai import OpenAI
from PIL import Image

PROMPTS = json.load(open("data/MAC/prompts.json"))

openai_client = OpenAI(api_key="")


def get_model_response(question_type, question, screen_representation):
    prompt = PROMPTS[question_type]
    formatted_prompt = prompt \
        .replace("<screen_representation>", screen_representation) \
        .replace("<question>", question)

    model_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        # model="gpt-4-turbo",
        temperature=0,
        messages=[{
            "role": "user",
            "content": formatted_prompt
        }],
    )

    answer = json.loads(model_response.choices[0].message.content)["answer"]
    return answer
