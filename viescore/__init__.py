# viescore/__init__.py

from .utils import (
    mllm_output_to_dict
)
import math
from . import vie_prompts

class VIEScore:
    def __init__(self, backbone="gpt4o", task="t2i", key_path=None) -> None:
        self.task = task
        self.backbone_name = backbone

        if self.task not in ["t2i", "tie", "t2v"]:
            raise ValueError("task must be either 't2i', 'tie', or 't2v'")

        # --- Model initialization (no changes here) ---
        if self.backbone_name == "gpt4o":
            from .mllm_tools.openai import GPT4o
            self.model = GPT4o(key_path)
        elif self.backbone_name == "gpt4v":
            from .mllm_tools.openai import GPT4v
            self.model = GPT4v(key_path)
        elif self.backbone_name == "gemini":
            from .mllm_tools.gemini import Gemini
            self.model = Gemini()
        elif self.backbone_name == "idefics2":
            from .mllm_tools.idefics2_eval import Idefics2
            self.model = Idefics2()
        elif self.backbone_name == "mantis":
            from .mllm_tools.mantis_idefics2_eval import Mantis
            self.model = Mantis()
        elif self.backbone_name == "minicpmv":
            from .mllm_tools.minicpmv_eval import MiniCPMV
            self.model = MiniCPMV()
        elif self.backbone_name == "qwenvl":
            from .mllm_tools.qwenvl import QwenVL
            self.model = QwenVL()
        else:
            raise NotImplementedError("backbone not supported")
        
        # --- Prompt preparation (no changes here) ---
        self.context = vie_prompts._context_no_delimit
        if self.task == "t2i":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_one_image_gen_rule, vie_prompts._prompts_0shot_t2i_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ])
        elif self.task == "tie":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ])
        elif self.task == "t2v":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_one_video_gen_rule, vie_prompts._prompts_0shot_t2v_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_t2v_rule_PQ])
    
    def _prepare_for_mllm(self, image_prompts):
        """Prepares image prompts for MLLM, handling url vs local path."""
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]
        if self.backbone_name in ['gpt4o', 'gpt4v']:
            self.model.use_encode = False if isinstance(image_prompts[0], str) else True
        return image_prompts

    def _get_parsed_dict(self, prompt):
        """Helper to get and parse MLLM output with retry logic."""
        parsed_dict = False
        tries = 0
        max_tries = 1
        while parsed_dict is False:
            tries += 1
            guess_if_cannot_parse = tries > max_tries
            result = self.model.get_parsed_output(prompt)
            parsed_dict = mllm_output_to_dict(result, give_up_parsing=guess_if_cannot_parse)
        
        if parsed_dict == "rate_limit_exceeded":
            print("rate_limit_exceeded") 
            raise ValueError("rate_limit_exceeded")
        return parsed_dict

    def evaluate_sc(self, image_prompts, text_prompt):
        """
        [NEW] Evaluates ONLY the SC (Semantic Consistency) score.
        """
        image_prompts = self._prepare_for_mllm(image_prompts)
        if self.task in ["t2i", "t2v"]:
            _SC_prompt = self.SC_prompt.replace("<prompt>", text_prompt)
        elif self.task == "tie":
            _SC_prompt = self.SC_prompt.replace("<instruction>", text_prompt)
        
        SC_prompt_final = self.model.prepare_prompt(image_prompts, _SC_prompt)
        return self._get_parsed_dict(SC_prompt_final)

    def evaluate_pq(self, image_prompts):
        """
        [NEW] Evaluates ONLY the PQ (Perceptual Quality) score.
        """
        image_prompts = self._prepare_for_mllm(image_prompts)
        PQ_prompt_final = self.model.prepare_prompt(image_prompts, self.PQ_prompt)
        return self._get_parsed_dict(PQ_prompt_final)

    def evaluate(self, image_prompts, text_prompt):
        """
        [MODIFIED] Main evaluation method.
        Now calls sc and pq methods and returns a dictionary with full details.
        """
        # 1. Evaluate SC
        SC_dict = self.evaluate_sc(image_prompts, text_prompt)

        # 2. Evaluate PQ
        # For TIE task, PQ is evaluated only on the final generated image
        if self.task == "tie":
            pq_image_prompts = image_prompts[-1] if isinstance(image_prompts, list) else image_prompts
        else:
            pq_image_prompts = image_prompts
        PQ_dict = self.evaluate_pq(pq_image_prompts)

        return {"SC": SC_dict, "PQ": PQ_dict}