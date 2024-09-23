from openai import OpenAI

class Qwen:
    def __init__(self, system_prompt, api_key):
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key)

    def process(self, user_input):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content