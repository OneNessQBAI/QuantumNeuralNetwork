class NomicEmbed:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def process(self, data):
        return f"NomicEmbed processed: {data}"