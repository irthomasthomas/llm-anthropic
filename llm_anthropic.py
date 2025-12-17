from anthropic import Anthropic, AsyncAnthropic
import llm
import click
import json
from pydantic import Field, field_validator, model_validator
from typing import Optional, List, Union

DEFAULT_THINKING_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0

# Global cache settings - these can be toggled via CLI
_cache_settings = {
    "system_cache": True,
    "prompt_cache": True,
}


def get_cache_settings():
    """Get current cache settings."""
    return _cache_settings.copy()


def set_cache_setting(key, value):
    """Set a cache setting."""
    _cache_settings[key] = value


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="anthropic")
    def anthropic_group():
        """Commands for Anthropic models."""
        pass

    @anthropic_group.group(name="cache")
    def cache_group():
        """Manage Anthropic prompt caching settings."""
        pass

    @cache_group.command(name="status")
    def cache_status():
        """Show current cache settings."""
        settings = get_cache_settings()
        click.echo("Anthropic Cache Settings:")
        click.echo(f"  System cache: {'enabled' if settings['system_cache'] else 'disabled'}")
        click.echo(f"  Prompt cache: {'enabled' if settings['prompt_cache'] else 'disabled'}")

    @cache_group.command(name="toggle-system")
    def toggle_system_cache():
        """Toggle system prompt caching on/off."""
        current = _cache_settings["system_cache"]
        _cache_settings["system_cache"] = not current
        new_state = "enabled" if _cache_settings["system_cache"] else "disabled"
        click.echo(f"System cache is now {new_state}")

    @cache_group.command(name="toggle-prompt")
    def toggle_prompt_cache():
        """Toggle user prompt caching on/off."""
        current = _cache_settings["prompt_cache"]
        _cache_settings["prompt_cache"] = not current
        new_state = "enabled" if _cache_settings["prompt_cache"] else "disabled"
        click.echo(f"Prompt cache is now {new_state}")

    @cache_group.command(name="toggle-all")
    def toggle_all_cache():
        """Toggle all caching on/off."""
        # If any cache is enabled, disable all; otherwise enable all
        any_enabled = any(_cache_settings.values())
        new_value = not any_enabled
        _cache_settings["system_cache"] = new_value
        _cache_settings["prompt_cache"] = new_value
        new_state = "enabled" if new_value else "disabled"
        click.echo(f"All caching is now {new_state}")

    @cache_group.command(name="enable-all")
    def enable_all_cache():
        """Enable all caching."""
        _cache_settings["system_cache"] = True
        _cache_settings["prompt_cache"] = True
        click.echo("All caching is now enabled")

    @cache_group.command(name="disable-all")
    def disable_all_cache():
        """Disable all caching."""
        _cache_settings["system_cache"] = False
        _cache_settings["prompt_cache"] = False
        click.echo("All caching is now disabled")


@llm.hookimpl
def register_models(register):
    # https://docs.anthropic.com/claude/docs/models-overview
    register(
        ClaudeMessages("claude-3-opus-20240229"),
        AsyncClaudeMessages("claude-3-opus-20240229"),
    )
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
        ClaudeMessages(
            "claude-3-5-sonnet-20240620", supports_pdf=True, default_max_tokens=8192
        ),
        AsyncClaudeMessages(
            "claude-3-5-sonnet-20240620", supports_pdf=True, default_max_tokens=8192
        ),
    )
    register(
        ClaudeMessages(
            "claude-3-5-sonnet-20241022", supports_pdf=True, default_max_tokens=8192
        ),
        AsyncClaudeMessages(
            "claude-3-5-sonnet-20241022", supports_pdf=True, default_max_tokens=8192
        ),
    )
    register(
        ClaudeMessages(
            "claude-3-5-sonnet-latest", supports_pdf=True, default_max_tokens=8192
        ),
        AsyncClaudeMessages(
            "claude-3-5-sonnet-latest", supports_pdf=True, default_max_tokens=8192
        ),
        aliases=("claude-3.5-sonnet", "claude-3.5-sonnet-latest"),
    )
    register(
        ClaudeMessages("claude-3-5-haiku-latest", default_max_tokens=8192),
        AsyncClaudeMessages("claude-3-5-haiku-latest", default_max_tokens=8192),
        aliases=("claude-3.5-haiku",),
    )
    # 3.7
    register(
        ClaudeMessages(
            "claude-3-7-sonnet-20250219",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-3-7-sonnet-20250219",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
    )
    register(
        ClaudeMessages(
            "claude-3-7-sonnet-latest",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-3-7-sonnet-latest",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-3.7-sonnet", "claude-3.7-sonnet-latest"),
    )
    # Claude 4.5 models
    register(
        ClaudeMessages(
            "claude-sonnet-4-5-20250929",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-sonnet-4-5-20250929",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-sonnet-4.5",),
    )
    register(
        ClaudeMessages(
            "claude-haiku-4-5-20251001",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-haiku-4-5-20251001",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-haiku-4.5",),
    )
    register(
        ClaudeMessages(
            "claude-opus-4-5-20251101",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-5-20251101",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-opus-4.5",),
    )
    # Claude 4 and 4.1 models
    register(
        ClaudeMessages(
            "claude-opus-4-20250514",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-20250514",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-opus-4",),
    )
    register(
        ClaudeMessages(
            "claude-opus-4-1-20250805",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        AsyncClaudeMessages(
            "claude-opus-4-1-20250805",
            supports_pdf=True,
            supports_thinking=True,
            default_max_tokens=8192,
        ),
        aliases=("claude-opus-4.1",),
    )


class ClaudeOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=None,
    )

    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=None,
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

    prefill: Optional[str] = Field(
        description="A prefill to use for the response",
        default=None,
    )

    hide_prefill: Optional[bool] = Field(
        description="Do not repeat the prefill value at the start of the response",
        default=None,
    )

    stop_sequences: Optional[Union[list, str]] = Field(
        description=(
            "Custom text sequences that will cause the model to stop generating - "
            "pass either a list of strings or a single string"
        ),
        default=None,
    )

    cache_prompt: Optional[bool] = Field(
        description="Whether to cache the user prompt for future use (defaults to global setting)",
        default=None,
    )

    cache_system: Optional[bool] = Field(
        description="Whether to cache the system prompt for future use (defaults to global setting)",
        default=None,
    )

    @field_validator("stop_sequences")
    def validate_stop_sequences(cls, stop_sequences):
        error_msg = "stop_sequences must be a list of strings or a single string"
        if isinstance(stop_sequences, str):
            try:
                stop_sequences = json.loads(stop_sequences)
                if not isinstance(stop_sequences, list) or not all(
                    isinstance(seq, str) for seq in stop_sequences
                ):
                    raise ValueError(error_msg)
                return stop_sequences
            except json.JSONDecodeError:
                return [stop_sequences]
        elif isinstance(stop_sequences, list):
            if not all(isinstance(seq, str) for seq in stop_sequences):
                raise ValueError(error_msg)
            return stop_sequences
        else:
            raise ValueError(error_msg)

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


class ClaudeOptionsWithThinking(ClaudeOptions):
    thinking: Optional[bool] = Field(
        description="Enable thinking mode",
        default=None,
    )
    thinking_budget: Optional[int] = Field(
        description="Number of tokens to budget for thinking", default=None
    )


class _Shared:
    needs_key = "anthropic"
    key_env_var = "ANTHROPIC_API_KEY"
    can_stream = True

    supports_thinking = False
    supports_schema = True
    default_max_tokens = 4096

    class Options(ClaudeOptions): ...

    def __init__(
        self,
        model_id,
        claude_model_id=None,
        supports_images=True,
        supports_pdf=False,
        supports_thinking=False,
        default_max_tokens=None,
    ):
        self.model_id = "anthropic/" + model_id
        self.claude_model_id = claude_model_id or model_id
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
        if supports_thinking:
            self.supports_thinking = True
            self.Options = ClaudeOptionsWithThinking
        if default_max_tokens is not None:
            self.default_max_tokens = default_max_tokens

    def prefill_text(self, prompt):
        if prompt.options.prefill and not prompt.options.hide_prefill:
            return prompt.options.prefill
        return ""

    def _should_cache_prompt(self, prompt):
        """Determine if prompt caching should be used."""
        if prompt.options.cache_prompt is not None:
            return prompt.options.cache_prompt
        return _cache_settings["prompt_cache"]

    def _should_cache_system(self, prompt):
        """Determine if system prompt caching should be used."""
        if prompt.options.cache_system is not None:
            return prompt.options.cache_system
        return _cache_settings["system_cache"]

    def build_messages(self, prompt, conversation) -> List[dict]:
        messages = []
        cache_control_count = 0
        max_cache_control_blocks = 2  # Leave one for the current prompt and system prompt
        cache_prompt = self._should_cache_prompt(prompt)

        if conversation:
            for response in reversed(conversation.responses):
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
                    
                    user_message = {
                        "role": "user",
                        "content": content,
                    }
                    
                    if cache_prompt and cache_control_count < max_cache_control_blocks:
                        user_message["content"][-1]["cache_control"] = {"type": "ephemeral"}
                        cache_control_count += 1
                    
                    messages.insert(0, user_message)
                else:
                    user_message = {
                        "role": "user",
                        "content": [{"type": "text", "text": response.prompt.prompt}],
                    }
                    
                    if cache_prompt and cache_control_count < max_cache_control_blocks:
                        user_message["content"][0]["cache_control"] = {"type": "ephemeral"}
                        cache_control_count += 1
                    
                    messages.insert(0, user_message)
                
                messages.insert(1, {"role": "assistant", "content": response.text_or_raise()})

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
            if prompt.prompt:
                text_content = {"type": "text", "text": prompt.prompt}
                if cache_prompt:
                    text_content["cache_control"] = {"type": "ephemeral"}
                content.append(text_content)
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        elif prompt.prompt:
            content = [{"type": "text", "text": prompt.prompt}]
            if cache_prompt:
                content[0]["cache_control"] = {"type": "ephemeral"}
            messages.append({"role": "user", "content": content})
            
        if prompt.options.prefill:
            messages.append({"role": "assistant", "content": prompt.options.prefill})
        return messages

    def build_kwargs(self, prompt, conversation):
        kwargs = {
            "model": self.claude_model_id,
            "messages": self.build_messages(prompt, conversation),
        }
        if prompt.options.user_id:
            kwargs["metadata"] = {"user_id": prompt.options.user_id}

        if prompt.options.top_p:
            kwargs["top_p"] = prompt.options.top_p
        else:
            kwargs["temperature"] = (
                prompt.options.temperature
                if prompt.options.temperature is not None
                else DEFAULT_TEMPERATURE
            )

        if prompt.options.top_k:
            kwargs["top_k"] = prompt.options.top_k

        if prompt.system:
            if self._should_cache_system(prompt):
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": prompt.system,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                kwargs["system"] = prompt.system

        if prompt.options.stop_sequences:
            kwargs["stop_sequences"] = prompt.options.stop_sequences

        if self.supports_thinking and (
            prompt.options.thinking or prompt.options.thinking_budget
        ):
            prompt.options.thinking = True
            budget_tokens = prompt.options.thinking_budget or DEFAULT_THINKING_TOKENS
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

        max_tokens = self.default_max_tokens
        if prompt.options.max_tokens is not None:
            max_tokens = prompt.options.max_tokens
        if (
            self.supports_thinking
            and prompt.options.thinking_budget is not None
            and prompt.options.thinking_budget > max_tokens
        ):
            max_tokens = prompt.options.thinking_budget + 1
        kwargs["max_tokens"] = max_tokens
        if max_tokens > 64000:
            kwargs["betas"] = ["output-128k-2025-02-19"]
            if "thinking" in kwargs:
                kwargs["extra_body"] = {"thinking": kwargs.pop("thinking")}

        if prompt.schema:
            kwargs["tools"] = [
                {
                    "name": "output_structured_data",
                    "input_schema": prompt.schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "output_structured_data"}

        return kwargs

    def set_usage(self, response):
        usage = response.response_json.get("usage")
        if usage:
            response.set_usage(
                input=usage.get("input_tokens"), output=usage.get("output_tokens")
            )

    def __str__(self):
        return "Anthropic Messages: {}".format(self.model_id)


class ClaudeMessages(_Shared, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key):
        client = Anthropic(
            api_key=self.get_key(key),
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        kwargs = self.build_kwargs(prompt, conversation)
        prefill_text = self.prefill_text(prompt)
        if "betas" in kwargs:
            messages_client = client.beta.messages
        else:
            messages_client = client.messages
        if stream:
            with messages_client.stream(**kwargs) as stream:
                if prefill_text:
                    yield prefill_text
                for chunk in stream:
                    if hasattr(chunk, "delta"):
                        delta = chunk.delta
                        if hasattr(delta, "text"):
                            yield delta.text
                        elif hasattr(delta, "partial_json"):
                            yield delta.partial_json
                # This records usage and other data:
                response.response_json = stream.get_final_message().model_dump()
        else:
            completion = messages_client.create(**kwargs)
            text = "".join(
                [item.text for item in completion.content if hasattr(item, "text")]
            )
            yield prefill_text + text
            response.response_json = completion.model_dump()
        self.set_usage(response)


class AsyncClaudeMessages(_Shared, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        client = AsyncAnthropic(
            api_key=self.get_key(key),
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        kwargs = self.build_kwargs(prompt, conversation)
        if "betas" in kwargs:
            messages_client = client.beta.messages
        else:
            messages_client = client.messages
        prefill_text = self.prefill_text(prompt)
        if stream:
            async with messages_client.stream(**kwargs) as stream_obj:
                if prefill_text:
                    yield prefill_text
                async for chunk in stream_obj:
                    if hasattr(chunk, "delta"):
                        delta = chunk.delta
                        if hasattr(delta, "text"):
                            yield delta.text
                        elif hasattr(delta, "partial_json"):
                            yield delta.partial_json
            response.response_json = (await stream_obj.get_final_message()).model_dump()
        else:
            completion = await messages_client.create(**kwargs)
            text = "".join(
                [item.text for item in completion.content if hasattr(item, "text")]
            )
            yield prefill_text + text
            response.response_json = completion.model_dump()
        self.set_usage(response)
