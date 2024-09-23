from openai import OpenAI

class Qwen2Math:
    def __init__(self, system_prompt, api_key):
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key)

    def compute(self, math_input):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Solve the following mathematical problem: {math_input}"}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content