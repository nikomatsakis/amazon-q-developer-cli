# MCP Sampling Protocol Flow

This document shows the complete message flow between Amazon Q CLI and an MCP server during sampling requests, based on the official MCP specification (2025-06-18).

## 1. MCP Sampling Request (Server-Initiated)

At some point after initialization, MCPs may initiate sampling. Sampling is started asynchronously and is not necessarily connected to tool execution.

### 1.1 Server → Client: Sampling Request

The MCP server initiates sampling by sending a sampling request with a particular id:

```json
{
  "jsonrpc": "2.0",
  "id": "sampling-123",
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "What is the capital of France?"
        }
      }
    ],
    "modelPreferences": {
      "hints": [
        {
          "name": "claude-3-sonnet"
        }
      ],
      "costPriority": 0.3,
      "speedPriority": 0.8,
      "intelligencePriority": 0.5
    },
    "systemPrompt": "You are a helpful assistant.",
    "includeContext": "none",
    "temperature": 0.7,
    "maxTokens": 100,
    "stopSequences": ["END"],
    "metadata": {
      "requestSource": "test-server"
    }
  }
}
```

**Key Parameters (per MCP spec):**
- `messages`: Array of `SamplingMessage` objects (required)
- `maxTokens`: Maximum tokens to sample (required)
- `modelPreferences`: Optional model selection hints and priorities
- `systemPrompt`: Optional system prompt (client MAY modify or omit)
- `includeContext`: Optional context inclusion ("none" | "thisServer" | "allServers")
- `temperature`: Optional temperature parameter
- `stopSequences`: Optional stop sequences
- `metadata`: Optional provider-specific metadata

## 2. User Approval Flow

Upon receiving the message, the client must ensure that a "human in the loop" has the ability to deny sampling requests.
In Q CLI, we use our trust system to decide whether sampling should be permitted freely or whether the user should be presented with the request.
If the user is presented with the request, they can also edit it before allowing it to continue.

## 3. Sampling Response

Once processing is complete, the response is sent back as a JSON RPC response message.
The `id` field corresponds to the id used for the sampling request.

### 3.1 Client → Server: Sampling Success Response

A successful response includes the `result` field:

```json
{
  "jsonrpc": "2.0",
  "id": "sampling-123",
  "result": {
    "role": "assistant",
    "content": {
      "type": "text", 
      "text": "The capital of France is Paris."
    },
    "model": "claude-3-sonnet-20240307",
    "stopReason": "endTurn"
  }
}
```

**Response Fields (per MCP spec):**
- `role`: Role of the response (typically "assistant")
- `content`: Content object (TextContent | ImageContent | AudioContent)
- `model`: Name of the model that generated the message (required)
- `stopReason`: Optional reason why sampling stopped

### 3.2 Alternative: Sampling Rejection Response

An error result includes the `error` field:

```json
{
  "jsonrpc": "2.0",
  "id": "sampling-123",
  "error": {
    "code": -1,
    "message": "User rejected sampling request"
  }
}
```

## Model Preferences System

The MCP spec defines a sophisticated model selection system:

### Capability Priorities (0-1 scale)
- `costPriority`: How important is minimizing costs?
- `speedPriority`: How important is low latency?
- `intelligencePriority`: How important are advanced capabilities?

### Model Hints
- Hints are treated as substrings that can match model names flexibly
- Multiple hints are evaluated in order of preference
- Clients **MAY** map hints to equivalent models from different providers
- Hints are advisory—clients make final model selection

Example mapping:
```json
{
  "hints": [
    { "name": "claude-3-sonnet" },  // Prefer Sonnet-class models
    { "name": "claude" }            // Fall back to any Claude model
  ],
  "costPriority": 0.3,      // Cost is less important
  "speedPriority": 0.8,     // Speed is very important
  "intelligencePriority": 0.5  // Moderate capability needs
}
```

## Content Types

The MCP spec supports multiple content types (not all of which are supported by our implementation):

### Text Content
```json
{
  "type": "text",
  "text": "The message content"
}
```

### Image Content
```json
{
  "type": "image",
  "data": "base64-encoded-image-data",
  "mimeType": "image/jpeg"
}
```

### Audio Content
```json
{
  "type": "audio",
  "data": "base64-encoded-audio-data",
  "mimeType": "audio/wav"
}
```

## Context Integration

The `includeContext` parameter allows servers to request context from MCP servers:

- `"none"`: No additional context (default)
- `"thisServer"`: Include context from the requesting server
- `"allServers"`: Include context from all connected MCP servers

**Note**: The client MAY ignore this request (and we currently do).

