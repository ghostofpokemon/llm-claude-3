from anthropic import Anthropic
import llm
from pydantic import Field, field_validator, model_validator
from typing import Optional, List, Union
import base64
import os
import mimetypes


@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/docs/models-overview
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
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=1.0,
    )

    top_p: Optional[float] = Field(
        description="Use nucleus sampling. In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by top_p. You should either alter temperature or top_p, but not both. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    top_k: Optional[int] = Field(
        description="Only sample from the top K options for each subsequent token. Used to remove 'long tail' low probability responses. Recommended for advanced use cases only. You usually only need to use temperature.",
        default=None,
    )

    user_id: Optional[str] = Field(
        description="An external identifier for the user who is associated with the request",
        default=None,
    )

    images: Optional[Union[str, List[str]]] = Field(
        description="Image file path(s) to be included in the request. Can be a comma-separated string or a list of strings.",
        default=None,
    )

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, max_tokens):
        real_max = cls.model_fields["max_tokens"].default
        if not (0 < max_tokens <= real_max):
            raise ValueError(f"max_tokens must be in range 1-{real_max}")
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

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self

    @field_validator("images")
    @classmethod
    def validate_images(cls, images):
        if images is None:
            return None

        if isinstance(images, str):
            # Split the string by commas or spaces
            image_paths = [path.strip() for path in images.replace(',', ' ').split()]
        elif isinstance(images, list):
            image_paths = images
        else:
            raise ValueError("images must be a string or a list of strings")

        validated_images = []
        for image_path in image_paths:
            expanded_path = os.path.expanduser(image_path)
            if not os.path.isfile(expanded_path):
                raise ValueError(f"Image file not found: {expanded_path}")
            validated_images.append(expanded_path)
        return validated_images


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
                # This records usage and other data:
                response.response_json = stream.get_final_message().model_dump()
        else:
            completion = client.messages.create(**kwargs)
            yield completion.content[0].text
            response.response_json = completion.model_dump()

    def __str__(self):
        return f"Anthropic Messages: {self.model_id}"


class ClaudeMessagesLong(ClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=4_096 * 2,
        )
