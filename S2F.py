from vllm_intervention import VLLMInterventionGenerator
class ad_reason_op():
    def __init__(self):
        self.client = VLLMInterventionGenerator(model_name="Qwen/Qwen3-8B")

    def response(self, messages):

        prompt_without_choices, options_text = messages[1]['content'], ""
        options_text = "Now I need to choose an answer based on my intuition" + options_text
        messages = [{'role': 'system', 'content': messages[0]['content']}, {'role': 'user', 'content': prompt_without_choices}]
        response = self.client.stream_response(messages, options_text, temperature=0, hesitation_threshold=3)
        return response               
