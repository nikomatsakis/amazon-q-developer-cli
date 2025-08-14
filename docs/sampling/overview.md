# Sampling

*Sampling* is a feature by which the MCP server can request that the client (i.e., Q CLI) perform completions on its behalf and send it the result.
The user is always empowered to review and edit any requests before they are sent to the LLM.

## Trust model

Sampling requests are initiated asynchronously by the MCP server and are not tied to any particular tool.
For the purposes of deciding trust, we treat sampling as a "pseudo-tool" (`@server/<sampling>`), which can be trusted or not the same as any other MCP tool.
This means that wildcards like `@server/*` will also permit sampling.

## 