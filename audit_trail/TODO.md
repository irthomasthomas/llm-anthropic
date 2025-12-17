# TODO: Future Cache Features

This document tracks potential future enhancements for the Anthropic prompt caching functionality.

## Planned Features (Not Yet Implemented)

### Cache Alert
**Description:** Notify the user before a cache is about to expire.

- Send a notification/alert when a cached prompt is nearing its expiration time
- Could be implemented as a warning message during prompts
- Useful for long-running conversations or expensive prompts

### Cache Keep-Alive
**Description:** Send a ping prompt before a cache expires to keep the cache hot.

- Automatically send minimal requests to refresh cache before expiration
- Configurable keep-alive time (in minutes)
- Stops when the specified keep-alive duration expires
- Useful for maintaining warm caches during interactive sessions
- Consider implementing as a background task or daemon

## Implementation Notes

- Anthropic's prompt caching has a 5-minute TTL by default
- Cache breakpoints are marked with `cache_control: {"type": "ephemeral"}`
- Maximum of 4 cache breakpoints per request
- Minimum cacheable prompt prefix is 1024 tokens (2048 for Claude 3.5 Haiku, 4096 for Claude 3 Opus)

## References

- [Anthropic Prompt Caching Documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
