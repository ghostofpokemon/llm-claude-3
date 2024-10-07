from anthropic import Anthropic
import llm
from pydantic import Field, field_validator, model_validator
from typing import Optional, List, Union
import base64
import os
import mimetypes

@llm.hookimpl
def register_models(register):
    register(ClaudeMessages("claude-3-opus-20240229"), aliases=("claude-3-opus",))
    register(ClaudeMessages("claude-3-sonnet-20240229"), aliases=("claude-3-sonnet",))
    register(ClaudeMessages("claude-3-haiku-20240307"), aliases=("claude-3-haiku",))
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-20240620"), aliases=("claude-3.5-sonnet",)
    )


class ClaudeOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=4_096,
    )
    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0.",
        default=1.0,
    )
    top_p: Optional[float] = Field(
        description="Use nucleus sampling. Recommended for advanced use cases only.",
        default=None,
    )
    top_k: Optional[int] = Field(
        description="Only sample from the top K options for each subsequent token.",
        default=None,
    )
    user_id: Optional[str] = Field(
        description="An external identifier for the user who is associated with the request",
        default=None,
    )
    images: Optional[Union[str, List[str]]] = Field(
        description="Image file path(s) to be included in the request",
        default=None,
    )

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, max_tokens):
        real_max = cls.model_fields["max_tokens"].default
        if not (0 < max_tokens <= real_max):
            raise ValueError("max_tokens must be in range 1-{}".format(real_max))
        return max_tokens

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, temperature):
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature must be in range 0.0-1.0")
        return temperature

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, top_p):
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be in range 0.0-1.0")
        return top_p

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, top_k):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k

    @field_validator("images")
    @classmethod
    def validate_images(cls, images):
        if images is None:
            return None
        if isinstance(images, str):
            images = [img.strip() for img in images.split(',')]
        for image_path in images:
            if not os.path.isfile(image_path):
                raise ValueError(f"Image file not found: {image_path}")
        return images

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self

class ClaudeMessages(llm.Model):
    needs_key = "claude"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    class Options(ClaudeOptions): ...

    def __init__(self, model_id, claude_model_id=None, extra_headers=None):
        self.model_id = model_id
        self.claude_model_id = claude_model_id or model_id
        self.extra_headers = extra_headers

    def process_images(self, image_paths):
        if not image_paths:
            return []

        processed_images = []
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type, _ = mimetypes.guess_type(image_path)
                if mime_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                    raise ValueError(f"Unsupported image type: {mime_type}")
                processed_images.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": image_data,
                    }
                })
        return processed_images

    def build_messages(self, prompt, conversation):
        messages = []
        if conversation:
            for response in conversation.responses:
                user_content = [{"type": "text", "text": response.prompt.prompt}]
                if response.prompt.options.images:
                    user_content.extend(self.process_images(response.prompt.options.images))
                messages.extend([
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response.text()},
                ])

        user_content = [{"type": "text", "text": prompt.prompt}]
        if prompt.options.images:
            user_content.extend(self.process_images(prompt.options.images))
        messages.append({"role": "user", "content": user_content})

        return messages

    def execute(self, prompt, stream, response, conversation):
        client = Anthropic(api_key=self.get_key())

        kwargs = {
            "model": self.claude_model_id,
            "messages": self.build_messages(prompt, conversation),
            "max_tokens": prompt.options.max_tokens,
        }
        if prompt.options.user_id:
            kwargs["metadata"] = {"user_id": prompt.options.user_id}

        if prompt.options.top_p:
            kwargs["top_p"] = prompt.options.top_p
        else:
            kwargs["temperature"] = prompt.options.temperature

        if prompt.options.top_k:
            kwargs["top_k"] = prompt.options.top_k

        if prompt.system:
            kwargs["system"] = prompt.system

        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        if stream:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                response.response_json = stream.get_final_message().model_dump()
        else:
            completion = client.messages.create(**kwargs)
            yield completion.content[0].text
            response.response_json = completion.model_dump()

    def __str__(self):
        return "Anthropic Messages: {}".format(self.model_id)


class ClaudeMessagesLong(ClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=4_096 * 2,
        )
