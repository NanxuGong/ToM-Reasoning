from vllm_intervention import VLLMInterventionGenerator
class ad_reason_3():
    def __init__(self):
        self.client = VLLMInterventionGenerator(model_name="Qwen/Qwen3-8B")

    def extract_and_remove_choices(self, user_prompt: str) -> tuple[str, str]:
        choices_index = user_prompt.find('choices: ')
        
        if choices_index != -1:
            prompt_without_choices = user_prompt[:choices_index].strip()
            choices = user_prompt[choices_index + len('choices: '):].strip()
            return prompt_without_choices, choices
        
        options_index = user_prompt.find('# Options')
        
        if options_index != -1:
            prompt_without_choices = user_prompt[:options_index].strip()
            choices = user_prompt[options_index + len('# Options'):].strip()
            return prompt_without_choices, choices

        candidates_index = user_prompt.find('[Candidate Answers]')
        
        if candidates_index != -1:
            prompt_without_choices = user_prompt[:candidates_index].strip()
            choices = user_prompt[candidates_index + len('[Candidate Answers]'):].strip()
            return prompt_without_choices, choices

        return None

    def response(self, messages):

        prompt_without_choices, options_text = self.extract_and_remove_choices(messages[1]['content'])
        options_text = "Now I need to choose an answer based on my intuition from: " + options_text
        messages = [{'role': 'system', 'content': messages[0]['content']}, {'role': 'user', 'content': prompt_without_choices}]
        response = self.client.stream_response(messages, options_text, temperature=0, hesitation_threshold=3)
        return response             