import re
import random
from utils_chat import chat_gpt4o
from utils import print_error, mk_pbar

ERROR_PROMPT = """
Can not find the index in the <index> </index> tags. Please reply with your answer in the <index> </index> tags.
"""
DEBUG_MODE = True

class VoteModel:
    def __init__(self, vote_prompt, error_prompt=ERROR_PROMPT, chat_func=chat_gpt4o):
        self.chat_func = chat_func
        self.vote_prompt = vote_prompt
        self.error_prompt = error_prompt
    
    def _get_single_vote(self, prompt, max_retry = 3, max_vote_idx = None):
        messages = None
        for _ in range(max_retry):
            response, messages = self.chat_func(prompt, messages, temperature=0.8)
            if DEBUG_MODE:
                print_error('[DEBUG] ' + response.replace('\n', '\\n'))
            
            pattern = r'<index>(.*?)</index>'
            match = re.search(pattern, response)
            if match:
                index = int(match.group(1))
            else:
                prompt = self.error_prompt
                continue

            if max_vote_idx is not None and index > max_vote_idx:
                prompt = f"""
OBSERVATION: The index is greater than the maximum vote index, which is {max_vote_idx}, but get {index}.
Please try again.
"""
                continue

            return index
        return None

    def vote(self, vote_times = 3, vote_prompt = None):
        vote_results = []
        for _ in mk_pbar(range(vote_times), desc="Voting"):
            vote_prompt = vote_prompt if vote_prompt is not None else self.vote_prompt
            response_idx = self._get_single_vote(vote_prompt)
            if response_idx is not None:
                vote_results.append(response_idx)
            else:
                print_error("Failed to get a valid vote result")

        # choose the most frequent result
        return max(set(vote_results), key=vote_results.count)


class ShuffleVoteModel(VoteModel):
    """The VoteModel that shuffle the inputs, in order to prevent positional bias in `ChatGPT` models."""
    def __init__(self, vote_prompt, error_prompt=ERROR_PROMPT, chat_func=chat_gpt4o):
        super().__init__(vote_prompt, error_prompt, chat_func)

    def __shuffle_inputs(self, inputs):
        img_str_list = inputs['img_str_list']

        random_indices = list(range(len(img_str_list)))
        random.shuffle(random_indices)
        # 打乱 img_str_list
        img_str_list = [img_str_list[i] for i in random_indices]
        # 建立映射关系，方便后续找回 vote 结果 index 的对应原始 index
        index_mapping = {i: random_indices[i] for i in range(len(random_indices))}

        return img_str_list, index_mapping
    
    def __format_inputs(self, inputs):
        generated_geometries = ""
        for idx, img_str in enumerate(inputs['img_str_list']):
            generated_geometries += f"""
{idx}. <img src='{img_str}'>
"""
        return self.vote_prompt.format(target_geometry=inputs['target_geometry'], generated_geometries=generated_geometries)

    def vote(self, inputs, vote_times = 3):
        vote_results = []
        for _ in mk_pbar(range(vote_times), desc="Voting"):
            img_str_list, index_mapping = self.__shuffle_inputs(inputs)
            shuffled_inputs = {
                'img_str_list': img_str_list,
                'target_geometry': inputs['target_geometry']
            }
            vote_prompt = self.__format_inputs(shuffled_inputs)
            response_idx = self._get_single_vote(vote_prompt, max_vote_idx=len(img_str_list) - 1)

            if response_idx is not None:
                response_idx = index_mapping[response_idx]
                if DEBUG_MODE:
                    print_error('[DEBUG] ' + f"Vote result: {response_idx}")
                vote_results.append(response_idx)
            else:
                print_error("Failed to get a valid vote result")
            

        final_index = max(set(vote_results), key=vote_results.count)
        if DEBUG_MODE:
            print_error(f"[DEBUG] Final Vote result: {final_index}")
        return final_index
