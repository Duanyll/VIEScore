import requests
import base64
import os
import mimetypes
from typing import List, Dict, Any

class QwenVL():
    def __init__(self, api_url: str = "http://192.168.5.101:10001/v1/chat/completions", model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct") -> None:
        """
        通过 VLLM 部署的 OpenAI 兼容 API 与 Qwen-VL 模型进行交互。

        requires: pip install requests
        Args:
            api_url (str): VLLM 聊天补全端点的 URL。
            model_name (str): API 请求中要使用的模型名称。
        """
        self.api_url = api_url
        self.model_name = model_name

    def _encode_image_to_data_uri(self, image_path: str) -> str:
        """将本地图像文件编码为 data URI。"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"在路径 {image_path} 未找到图像文件")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image'):
            mime_type = 'image/jpeg'  # 如果无法确定，则默认为 jpeg

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return f"data:{mime_type};base64,{encoded_string}"

    def prepare_prompt(self, image_links: List[str] = [], text_prompt: str = "") -> List[Dict[str, Any]]:
        """
        为 OpenAI 兼容 API 准备内容列表。

        Args:
            image_links (List[str]): 图像 URL 或本地文件路径的列表。
            text_prompt (str): 文本提示。

        Returns:
            List[Dict[str, Any]]: 用于 API 调用的内容列表。
        """
        if isinstance(image_links, str):
            image_links = [image_links]

        content_list = []

        # 首先处理并添加图像
        for image_link in image_links:
            # 判断是网络链接还是本地路径
            if image_link.startswith(('http://', 'https://')):
                final_image_url = image_link
            else:
                final_image_url = self._encode_image_to_data_uri(image_link)
            
            content_list.append({
                "type": "image_url",
                "image_url": {"url": final_image_url}
            })

        # 最后添加文本
        content_list.append({
            "type": "text",
            "text": text_prompt
        })

        return content_list

    def get_parsed_output(self, prepared_content: List[Dict[str, Any]]) -> str:
        """
        将准备好的内容发送到 VLLM API 并获取响应。

        Args:
            prepared_content (List[Dict[str, Any]]): 从 prepare_prompt 方法获取的内容列表。

        Returns:
            str: 来自模型的文本响应。
        """
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prepared_content
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()  # 如果状态码是 4xx 或 5xx，则抛出异常
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            return f"API 请求错误: {e}"
        except (KeyError, IndexError) as e:
            return f"解析 API 响应失败: {e}. 完整响应: {response.text}"

# __del__ 方法不再需要，因为我们不在磁盘上创建临时文件。

if __name__ == "__main__":
    # --- 使用示例 ---
    
    # 初始化客户端，指向您的 VLLM 服务
    # 您可以根据需要更改 api_url 和 model_name
    qwen_client = QwenVL(api_url="http://192.168.5.101:10001/v1/chat/completions")

    # 定义图片源：一个本地路径和一个网络 URL
    # 注意：请确保 'Walking_tiger_female.jpg' 文件存在于您的本地目录中，
    # 或者将其替换为您系统上的有效图片路径。
    local_image = '../Walking_tiger_female.jpg' # 这是一个示例路径
    if not os.path.exists(local_image):
         print(f"警告: 示例图片路径 '{local_image}' 不存在。请替换为有效路径以测试本地图片功能。")
         image_sources = [
             'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'
         ]
         text_prompt = 'What is in this image?'
    else:
        image_sources = [
            local_image,
            'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'
        ]
        text_prompt = 'What is the difference between these two images?'

    # 1. 准备 API 请求所需格式的 prompt
    print("正在准备 prompt...")
    prepared_prompt = qwen_client.prepare_prompt(image_sources, text_prompt)

    # 2. 发送请求并获取解析后的输出
    print("正在向 API 发送请求...")
    response_text = qwen_client.get_parsed_output(prepared_prompt)

    # 3. 打印模型的响应
    print("\n--- 模型响应 ---")
    print(response_text)
    
    # 预期的输出格式（与您提供的示例相似）:
    """
    The two images show two different animals in different environments. The first image shows a tiger walking through a grassy field, while the second image shows a zebra grazing in the grass. The tiger and the zebra have different physical characteristics, such as the tiger's stripes and the zebra's stripes, as well as different behavioral characteristics, such as the tiger's solitary hunting style and the zebra's social grazing habits.
    """