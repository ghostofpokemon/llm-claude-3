from anthropic import Anthropic, AsyncAnthropic
import llm
from pydantic import Field, field_validator, model_validator
from typing import Optional, List


@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/docs/models-overview
    register(
        ClaudeMessages("claude-3-opus-20240229"),
        AsyncClaudeMessages("claude-3-opus-20240229"),
    ),
    register(
        ClaudeMessages("claude-3-opus-latest"),
        AsyncClaudeMessages("claude-3-opus-latest"),
        aliases=("claude-3-opus",),
    )
    register(
        ClaudeMessages("claude-3-sonnet-20240229"),
        AsyncClaudeMessages("claude-3-sonnet-20240229"),
        aliases=("claude-3-sonnet",),
    )
    register(
        ClaudeMessages("claude-3-haiku-20240307"),
        AsyncClaudeMessages("claude-3-haiku-20240307"),
        aliases=("claude-3-haiku",),
    )
    # 3.5 models
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-20240620", supports_pdf=True),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-20240620", supports_pdf=True),
    )
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-20241022", supports_pdf=True),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-20241022", supports_pdf=True),
    )
    register(
        ClaudeMessagesLong("claude-3-5-sonnet-latest", supports_pdf=True),
        AsyncClaudeMessagesLong("claude-3-5-sonnet-latest", supports_pdf=True),
        aliases=("claude-3.5-sonnet", "claude-3.5-sonnet-latest"),
    )
    register(
        ClaudeMessagesLong("claude-3-5-haiku-latest", supports_images=False),
        AsyncClaudeMessagesLong("claude-3-5-haiku-latest", supports_images=False),
        aliases=("claude-3.5-haiku",),
    )
    # 3.7 models
    register(
        Claude37MessagesLong("claude-3-7-sonnet-20250219", supports_pdf=True),
        AsyncClaude37MessagesLong("claude-3-7-sonnet-20250219", supports_pdf=True),
    )
    register(
        Claude37MessagesLong("claude-3-7-sonnet-latest", supports_pdf=True),
        AsyncClaude37MessagesLong("claude-3-7-sonnet-latest", supports_pdf=True),
        aliases=("claude-3.7-sonnet", "claude-3.7-sonnet-latest", "claude-3.7"),
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

    @model_validator(mode="after")
    def validate_temperature_top_p(self):
        if self.temperature != 1.0 and self.top_p is not None:
            raise ValueError("Only one of temperature and top_p can be set")
        return self


class Claude37Options(ClaudeOptions):
    thinking: Optional[int] = Field(
        description="The budget of tokens to allocate for extended thinking (must be less than max_tokens). Range from 1024 to 63999. Default is None (disabled). When enabled, a value of 16000 is recommended for most use cases. Extended thinking gives Claude 3.7 enhanced reasoning capabilities for complex tasks.",
        default=None,
    )
    
    @field_validator("thinking")
    @classmethod
    def validate_thinking(cls, thinking):
        if thinking is not None and not (1024 <= thinking <= 63999):
            raise ValueError("thinking must be in range 1024-63999")
        return thinking
    
    @model_validator(mode="after")
    def validate_thinking_max_tokens(self):
        # Ensure thinking budget is less than max_tokens as per documentation
        if self.thinking is not None and self.thinking >= self.max_tokens:
            raise ValueError("thinking budget_tokens must be less than max_tokens")
        return self


long_field = Field(
    description="The maximum number of tokens to generate before stopping",
    default=4_096 * 2,
)

# Claude 3.7 has higher token limits
long_field_3_7 = Field(
    description="The maximum number of tokens to generate before stopping (up to 64K for Claude 3.7)",
    default=20_000,
)


class _Shared:
    needs_key = "claude"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    class Options(ClaudeOptions): ...

    def __init__(
        self,
        model_id,
        claude_model_id=None,
        extra_headers=None,
        supports_images=True,
        supports_pdf=False,
    ):
        self.model_id = model_id
        self.claude_model_id = claude_model_id or model_id
        self.extra_headers = extra_headers or {}
        if supports_pdf:
            self.extra_headers["anthropic-beta"] = "pdfs-2024-09-25"
        # Add extended output capability for Claude 3.7 models
        if "claude-3-7" in model_id:
            if "anthropic-beta" in self.extra_headers:
                self.extra_headers["anthropic-beta"] += ",output-128k-2025-02-19"
            else:
                self.extra_headers["anthropic-beta"] = "output-128k-2025-02-19"
        self.attachment_types = set()
        if supports_images:
            self.attachment_types.update(
                {
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    "image/gif",
                }
            )
        if supports_pdf:
            self.attachment_types.add("application/pdf")

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                if response.attachments:
                    content = [
                        {
                            "type": (
                                "document"
                                if attachment.resolve_type() == "application/pdf"
                                else "image"
                            ),
                            "source": {
                                "data": attachment.base64_content(),
                                "media_type": attachment.resolve_type(),
                                "type": "base64",
                            },
                        }
                        for attachment in response.attachments
                    ]
                    content.append({"type": "text", "text": response.prompt.prompt})
                else:
                    content = response.prompt.prompt
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": content,
                        },
                        {"role": "assistant", "content": response.text()},
                    ]
                )
        if prompt.attachments:
            content = [
                {
                    "type": (
                        "document"
                        if attachment.resolve_type() == "application/pdf"
                        else "image"
                    ),
                    "source": {
                        "data": attachment.base64_content(),
                        "media_type": attachment.resolve_type(),
                        "type": "base64",
                    },
                }
                for attachment in prompt.attachments
            ]
            content.append({"type": "text", "text": prompt.prompt})
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        else:
            messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def build_kwargs(self, prompt, conversation):
        kwargs = {
            "model": self.claude_model_id,
            "messages": self.build_messages(prompt, conversation),
            "max_tokens": prompt.options.max_tokens,
        }
        if prompt.options.user_id:
            kwargs["metadata"] = {"user_id": prompt.options.user_id}

        # Extended thinking is not compatible with temperature, top_p, or top_k modifications
        # as per the documentation
        if hasattr(prompt.options, "thinking") and prompt.options.thinking is not None:
            # Store thinking parameter separately - we'll handle it in the execute methods
            self.thinking_param = {
                "type": "enabled",
                "budget_tokens": prompt.options.thinking
            }
            # Don't add to kwargs here - we'll handle it in execute methods
            
            # Use default temperature when thinking is enabled
            kwargs["temperature"] = 1.0
        else:
            self.thinking_param = None
            # Use specified temperature/top_p/top_k when thinking is not enabled
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
        return kwargs

    def set_usage(self, response):
        usage = response.response_json.pop("usage")
        if usage:
            response.set_usage(
                input=usage.get("input_tokens"), output=usage.get("output_tokens")
            )

    def __str__(self):
        return "Anthropic Messages: {}".format(self.model_id)


class ClaudeMessages(_Shared, llm.Model):

    def execute(self, prompt, stream, response, conversation):
        client = Anthropic(api_key=self.get_key())
        kwargs = self.build_kwargs(prompt, conversation)
        
        # Add thinking parameter if we're using a Claude 3.7 model and thinking is enabled
        if hasattr(self, 'thinking_param') and self.thinking_param and "claude-3-7" in self.claude_model_id:
            try:
                # Try to import the latest version of anthropic to check if thinking is supported
                import importlib
                import inspect
                anthropic = importlib.import_module("anthropic")
                
                # Create a temporary client instance just for inspection
                inspection_client = anthropic.Anthropic(api_key="dummy_key")
                
                # Check if the client instance has messages attribute
                if hasattr(inspection_client, "messages") and hasattr(inspection_client.messages, "create"):
                    # Check if the client supports the thinking parameter
                    client_signature = inspect.signature(inspection_client.messages.create)
                    if "thinking" in client_signature.parameters:
                        kwargs["thinking"] = self.thinking_param
                    else:
                        import warnings
                        warnings.warn(
                            "The 'thinking' parameter was specified but is not supported by your "
                            "version of the Anthropic library. Please run 'pipx runpip llm install --upgrade anthropic' "
                            "to upgrade to the latest version to use this feature with Claude 3.7 models."
                        )
                else:
                    import warnings
                    warnings.warn(
                        "The 'thinking' parameter was specified but could not be applied. "
                        "The messages.create method was not found in the Anthropic client. "
                        "Please run 'pipx runpip llm install --upgrade anthropic' to upgrade."
                    )
            except (ImportError, AttributeError, TypeError) as e:
                # If we can't check or the parameter isn't supported, don't add it
                import warnings
                warnings.warn(
                    f"The 'thinking' parameter was specified but could not be applied. "
                    f"Error: {e}. This may be due to an outdated version of the Anthropic library. "
                    "Please run 'pipx runpip llm install --upgrade anthropic' to upgrade."
                )
        
        if stream:
            with client.messages.stream(**kwargs) as stream:
                # For streaming, we need to check each block as it comes in
                thinking_content = []
                in_thinking_block = False
                
                for chunk in stream:
                    # Look for content blocks
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "thinking"):
                        in_thinking_block = True
                        # We're in a thinking block
                        if chunk.delta.thinking:
                            thinking_content.append(chunk.delta.thinking)
                            yield f"{chunk.delta.thinking}"
                    elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text") and in_thinking_block:
                        # We've transitioned from thinking to text - yield a separator
                        yield "\n\n=== END OF THINKING | FINAL RESPONSE ===\n\n"
                        in_thinking_block = False
                        yield chunk.delta.text
                    elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                        # Normal text chunk
                        yield chunk.delta.text
                
                # This records usage and other data:
                response.response_json = stream.get_final_message().model_dump()
                if thinking_content:
                    response.thinking = "".join(thinking_content)
        else:
            completion = client.messages.create(**kwargs)
            
            # Check if there's thinking content in the response
            thinking_found = False
            thinking_content = None
            response_text = None
            
            if hasattr(completion, "content") and isinstance(completion.content, list):
                for item in completion.content:
                    if hasattr(item, "type"):
                        # Look for thinking content
                        if item.type == "thinking" and hasattr(item, "thinking"):
                            thinking_found = True
                            thinking_content = item.thinking
                        # Look for the response text
                        elif item.type == "text" and hasattr(item, "text"):
                            response_text = item.text
            
            # If thinking was found, yield it first
            if thinking_found and thinking_content:
                # Store thinking in the response object
                response.thinking = thinking_content
                # Yield the thinking content
                yield "=== CLAUDE'S THINKING PROCESS ===\n\n"
                yield thinking_content
                yield "\n\n=== FINAL RESPONSE ===\n\n"
                
            # Yield the response text
            if response_text:
                yield response_text
            else:
                # Fallback if we couldn't extract text properly
                yield completion.content[0].text
                
            response.response_json = completion.model_dump()
        self.set_usage(response)


class ClaudeMessagesLong(ClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = long_field


class Claude37MessagesLong(ClaudeMessages):
    class Options(Claude37Options):
        max_tokens: Optional[int] = long_field_3_7


class AsyncClaudeMessages(_Shared, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        client = AsyncAnthropic(api_key=self.get_key())
        kwargs = self.build_kwargs(prompt, conversation)
        
        # Add thinking parameter if we're using a Claude 3.7 model and thinking is enabled
        if hasattr(self, 'thinking_param') and self.thinking_param and "claude-3-7" in self.claude_model_id:
            try:
                # Try to import the latest version of anthropic to check if thinking is supported
                import importlib
                import inspect
                anthropic = importlib.import_module("anthropic")
                
                # Create a temporary client instance just for inspection
                inspection_client = anthropic.Anthropic(api_key="dummy_key")
                
                # Check if the client instance has messages attribute
                if hasattr(inspection_client, "messages") and hasattr(inspection_client.messages, "create"):
                    # Check if the client supports the thinking parameter
                    client_signature = inspect.signature(inspection_client.messages.create)
                    if "thinking" in client_signature.parameters:
                        kwargs["thinking"] = self.thinking_param
                    else:
                        import warnings
                        warnings.warn(
                            "The 'thinking' parameter was specified but is not supported by your "
                            "version of the Anthropic library. Please run 'pipx runpip llm install --upgrade anthropic' "
                            "to upgrade to the latest version to use this feature with Claude 3.7 models."
                        )
                else:
                    import warnings
                    warnings.warn(
                        "The 'thinking' parameter was specified but could not be applied. "
                        "The messages.create method was not found in the Anthropic client. "
                        "Please run 'pipx runpip llm install --upgrade anthropic' to upgrade."
                    )
            except (ImportError, AttributeError, TypeError) as e:
                # If we can't check or the parameter isn't supported, don't add it
                import warnings
                warnings.warn(
                    f"The 'thinking' parameter was specified but could not be applied. "
                    f"Error: {e}. This may be due to an outdated version of the Anthropic library. "
                    "Please run 'pipx runpip llm install --upgrade anthropic' to upgrade."
                )
                
        if stream:
            async with client.messages.stream(**kwargs) as stream_obj:
                # For streaming, we need to check each block as it comes in
                thinking_content = []
                in_thinking_block = False
                
                async for chunk in stream_obj:
                    # Look for content blocks
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "thinking"):
                        in_thinking_block = True
                        # We're in a thinking block
                        if chunk.delta.thinking:
                            thinking_content.append(chunk.delta.thinking)
                            yield f"{chunk.delta.thinking}"
                    elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text") and in_thinking_block:
                        # We've transitioned from thinking to text - yield a separator
                        yield "\n\n=== END OF THINKING | FINAL RESPONSE ===\n\n"
                        in_thinking_block = False
                        yield chunk.delta.text
                    elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                        # Normal text chunk
                        yield chunk.delta.text
                
                # This records usage and other data:
                response.response_json = (await stream_obj.get_final_message()).model_dump()
                if thinking_content:
                    response.thinking = "".join(thinking_content)
        else:
            completion = await client.messages.create(**kwargs)
            
            # Check if there's thinking content in the response
            thinking_found = False
            thinking_content = None
            response_text = None
            
            if hasattr(completion, "content") and isinstance(completion.content, list):
                for item in completion.content:
                    if hasattr(item, "type"):
                        # Look for thinking content
                        if item.type == "thinking" and hasattr(item, "thinking"):
                            thinking_found = True
                            thinking_content = item.thinking
                        # Look for the response text
                        elif item.type == "text" and hasattr(item, "text"):
                            response_text = item.text
            
            # If thinking was found, yield it first
            if thinking_found and thinking_content:
                # Store thinking in the response object
                response.thinking = thinking_content
                # Yield the thinking content
                yield "=== CLAUDE'S THINKING PROCESS ===\n\n"
                yield thinking_content
                yield "\n\n=== FINAL RESPONSE ===\n\n"
                
            # Yield the response text
            if response_text:
                yield response_text
            else:
                # Fallback if we couldn't extract text properly
                yield completion.content[0].text
                
            response.response_json = completion.model_dump()
        self.set_usage(response)


class AsyncClaudeMessagesLong(AsyncClaudeMessages):
    class Options(ClaudeOptions):
        max_tokens: Optional[int] = long_field


class AsyncClaude37MessagesLong(AsyncClaudeMessages):
    class Options(Claude37Options):
        max_tokens: Optional[int] = long_field_3_7
