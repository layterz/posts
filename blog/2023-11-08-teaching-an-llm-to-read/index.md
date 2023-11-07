---
slug: teaching-an-llm-to-read
title: Teaching an LLM to Read
---

If you’ve played around with language models for long enough you’ve probably run into something like

> The message you submitted was 8243 and the maximum context length is 4024. Please reduce the length of your input and try again.

For standard ChatGPT this is 4024 tokens, going up to 32,000 on OpenAI’s most expensive model. Even if your prompt fits within the context limit, output tends to be worse as input length grows. Or, more accurately, as the diversity of the input grows the output quality tends to lower. This mirrors how humans consume information — so, in my opinion, this points to something more fundamental than merely the limits of early AI systems.

So what can you do if your input can't fit into the context limit? In a [previous post](/generating-narratives), I introduced [promptx](https://github.com/layterz/promptx) and some techniques for combining prompts to generate complex output — a screenplay from a logline in that case. In this post I want to focus on some techniques for the reverse — i.e. consuming complex content across multiple prompts.