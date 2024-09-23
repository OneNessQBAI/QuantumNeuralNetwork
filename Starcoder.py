from openai import OpenAI

class Starcoder:
    def __init__(self, system_prompt, api_key):
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key)

    def optimize_code(self, code_input):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Optimize the following code:\n\n{code_input}"}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content