from openai import OpenAI

class DeepSeekCoder:
    def __init__(self, system_prompt, api_key):
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key)

    def generate_code(self, coder_input):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": coder_input}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content

    def select_best_response(self, responses):
        combined_responses = "\n".join(responses)
        selection_prompt = f"Analyze the following responses and select the most appropriate one:\n{combined_responses}"
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": selection_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content