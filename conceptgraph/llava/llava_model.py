import sys
import warnings

LLAVA_PYTHON_PATH = "/home/kev/packages/LLaVA-NeXT"
sys.path.append(LLAVA_PYTHON_PATH)

import llava.conversation


warnings.filterwarnings("ignore")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init

from transformers import CLIPVisionModel

from PIL import Image
import requests
import copy
import torch


class LLaVaNeXTChat:
    def __init__(
        self,
        model,
        image_processor,
        tokenizer,
        max_length,
        conv_template="llava_llama3",
    ):
        self.model = model
        self.vision_tower = CLIPVisionModel.from_pretrained(self.model.config.mm_vision_tower).to("cuda")
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.max_length = max_length
        self.device = model.device

        self.conv = copy.deepcopy(conv_templates[conv_template])
        self.conv.tokenizer = tokenizer # kev added so it doesn't throw an error
        self.pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

    def __call__(self, prompt, img=None):
        
        if img is not None:
            img_tensor = self.process_img(img=img)
            prompt = DEFAULT_IMAGE_TOKEN + f"\n{prompt}"
            img_sizes = [img.size]
            
        else:
            prompt = prompt
            img_tensor = None
            img_sizes = None
            
        self.conv.append_message(self.conv.roles[0], prompt)
        self.conv.append_message(self.conv.roles[1], None)
        
        
        prompt_question = self.conv.get_prompt()
        input_ids = (tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device))
        
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_ids).to(self.device)
        attention_masks = input_ids.ne(self.pad_token_ids).to(self.device)
        
        if img is not None:
            cont = self.model.generate(
                input_ids,
                images=img_tensor,
                image_sizes=img_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=256,
                pad_token_id=self.pad_token_ids,
                attention_mask=attention_masks
            )
        else:
            cont = self.model.generate(
                input_ids,
                images=img_tensor,
                image_sizes=img_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=256,
                pad_token_id=self.pad_token_ids,
                attention_mask=attention_masks
            )
            
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        
        return text_outputs
    
    def compute_image_features(self, img):
        image_forward_outs = self.vision_tower(img.to(self.vision_tower.dtype), output_hidden_states=True)
        select_hidden_state_layer = getattr(self.model.config, "mm_vision_select_layer", -2)
        select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
        
        # TODO (Krishna): Figure out why the zero-th dim is dropped (perhaps that is the global token?)
        image_features = select_hidden_state[:, 1:].to(img.dtype)
        # image_features = self.mm_projector(image_features)
        return image_features
        
    def process_img(self, img):
        
        img_tensor = process_images([img], self.image_processor, self.model.config)
        img_tensor = [_img.to(dtype=torch.float16, device=self.device) for _img in img_tensor]
        
        return img_tensor
        
    def reset(self):
        self.conv = copy.deepcopy(conv_templates[self.conv_template])
        self.conv.tokenizer = self.tokenizer # kev added so it doesn't throw an error
        
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IGNORE_INDEX,
    )
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init

    from PIL import Image
    import requests
    import copy
    import torch

    pretrained = "/home/kev/pretrained_models/llama3-llava-next-8b"
    model_name = "llava_llama3"
    device = "cuda"
    conv_template = (
        "llava_llama_3"  # Make sure you use correct chat template for different models
    )
    # device_map = "auto"
    device_map = 0
    
    image = Image.open("/home/kev/repos/concept-graphs/scannet.jpeg").convert("RGB")

    disable_torch_init()
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained,
        None,
        model_name,
        device_map=device_map,
        load_8bit=True,
        attn_implementation=None,
    )  # Add any other thing you want to pass in llava_model_args

    chat = LLaVaNeXTChat(
        model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_length=max_length,
        conv_template=conv_template,
    )
    chat.model.eval()
    chat.model.tie_weights()
    
    print("LLaVA chat initialized...")

    # url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    # image = Image.open(requests.get(url, stream=True).raw)

    # prompt = "List the set of objects in this image."
    # output = chat(prompt=prompt, img=image)
    # print(output)
    
    image_tensor = chat.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    print(image_tensor)
    print(image_tensor.shape)

    image_features = chat.encode_image(image_tensor[None, ...].half().cuda())
    
    print(image_features)
    
    # prompt = "Tell me a joke"
    # output = chat(prompt=prompt, img=None)
    # print(output)
    
    # prompt = "Explain it to me in more detail"
    # output = chat(prompt=prompt, img=None)
    # print(output)

    # prompt = "What did you say?"
    # output = chat(prompt=prompt, img=None)
    # print(output)
    
    # chat.reset()
    
    # prompt = "What did I ask you before and what did you reply? Summarize without giving me the exact words"
    # output = chat(prompt=prompt, img=None)
    # print(output)