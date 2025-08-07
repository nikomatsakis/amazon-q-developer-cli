use serde::{Deserialize, Serialize};

// Message types for MCP sampling communication between actors.
//
// *Convention:* the `Mcp` prefix indicates the type corresponds to the transport
// representation from the MCP spec. In that case, references to the spec
// are included in the doc comment.

/// Repackaged sampling request for internal Q CLI actor communication
///
/// This is Q CLI's internal representation of an MCP `sampling/createMessage` request,
/// parsed and enriched with routing information for communication between MCP Server
/// Actors and the UI Actor.
///
/// # Related types
///
/// * [`SamplingResponse`][] represents the internal representation of a response.
/// * [`McpSamplingCreateMessageRequestParams`][] represents the transport format.
/// * [`McpSamplingCreateMessageResult`][] represents the transport response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingRequest {
    pub server_name: String,
    pub request_id: String,
    pub messages: Vec<McpSamplingMessage>,
    pub model_preferences: Option<McpModelPreferences>,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
    // Note: includeContext parameter excluded for Phase 1
}

/// Individual message in a sampling request
///
/// Conforms to the MCP specification's `SamplingMessage` interface:
///
/// ```typescript
/// interface SamplingMessage {
///   content: TextContent | ImageContent | AudioContent;
///   role: Role;
/// }
/// ```
///
/// Reference: [MCP specification](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-06-18/schema.mdx#samplingmessage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingMessage {
    pub role: String,
    pub content: McpSamplingContent,
}

/// Content of a sampling message (Phase 1: text only)
///
/// Conforms to the MCP specification's `TextContent` interface:
///
/// ```typescript
/// interface TextContent {
///   type: "text";
///   text: string;
///   annotations?: Annotations;
///   _meta?: { [key: string]: unknown };
/// }
/// ```
///
/// Reference: [MCP specification](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-06-18/schema.mdx#textcontent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// Model preferences from MCP sampling request
///
/// Conforms to the MCP specification's `ModelPreferences` interface:
///
/// ```typescript
/// interface ModelPreferences {
///   costPriority?: number;
///   hints?: ModelHint[];
///   intelligencePriority?: number;
///   speedPriority?: number;
/// }
/// ```
///
/// Reference: [MCP specification](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-06-18/schema.mdx#modelpreferences)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpModelPreferences {
    pub hints: Option<Vec<McpModelHint>>,
    #[serde(rename = "costPriority")]
    pub cost_priority: Option<f64>,
    #[serde(rename = "speedPriority")]
    pub speed_priority: Option<f64>,
    #[serde(rename = "intelligencePriority")]
    pub intelligence_priority: Option<f64>,
}

/// Model hint for preference matching
///
/// Conforms to the MCP specification's `ModelHint` interface:
///
/// ```typescript
/// interface ModelHint {
///   name?: string;
/// }
/// ```
///
/// Reference: [MCP specification](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-06-18/schema.mdx#modelhint)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpModelHint {
    pub name: Option<String>,
}

/// Internal response from UI Actor back to MCP Server Actor
///
/// This is Q CLI's internal representation for communicating sampling results
/// between actors. The actual MCP protocol response uses [`McpSamplingCreateMessageResult`][]
/// format for successful responses, or JSON-RPC errors for rejections.
#[derive(Debug, Clone)]
pub enum SamplingResponse {
    Approved { request_id: String, llm_response: String },
    Rejected { request_id: String, reason: String },
}

/// Internal user approval decision for sampling requests
///
/// This represents the user's choice when presented with a sampling approval dialog.
/// Used internally by Q CLI's UI Actor to determine how to respond to sampling requests.
#[derive(Debug, Clone)]
pub enum SamplingApproval {
    /// User approved the sampling request (may or may not have been edited)
    Approve(SamplingRequest),
    /// User rejected the sampling request - will send an error response back to MCP server
    Reject,
    /// User chose to trust this server for all future sampling requests.
    /// The current request will be approved, and the server will be added to the trusted list
    /// so future sampling requests from this server are auto-approved without user interaction.
    TrustServer,
}

/// MCP sampling request parameters (for serde deserialization)
///
/// Conforms to the MCP specification's `CreateMessageRequest.params` interface:
///
/// ```typescript
/// interface CreateMessageRequest {
///   method: "sampling/createMessage";
///   params: {
///     includeContext?: "none" | "thisServer" | "allServers";
///     maxTokens: number;
///     messages: SamplingMessage[];
///     metadata?: object;
///     modelPreferences?: ModelPreferences;
///     stopSequences?: string[];
///     systemPrompt?: string;
///     temperature?: number;
///   };
/// }
/// ```
///
/// Reference: [MCP specification](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-06-18/schema.mdx#createmessagerequest)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpSamplingCreateMessageRequestParams {
    pub messages: Vec<McpSamplingMessage>,
    #[serde(rename = "modelPreferences")]
    pub model_preferences: Option<McpModelPreferences>,
    #[serde(rename = "systemPrompt")]
    pub system_prompt: Option<String>,
    #[serde(rename = "maxTokens")]
    pub max_tokens: Option<u32>,
}

/// Parse sampling request from MCP JSON-RPC parameters
pub fn parse_sampling_request(
    server_name: String,
    request_id: String,
    params: Option<serde_json::Value>,
) -> Result<SamplingRequest, String> {
    let params = params.ok_or_else(|| "Missing parameters for sampling request".to_string())?;

    let sampling_params: McpSamplingCreateMessageRequestParams =
        serde_json::from_value(params).map_err(|e| format!("Failed to parse sampling parameters: {}", e))?;

    Ok(SamplingRequest {
        server_name,
        request_id,
        messages: sampling_params.messages,
        model_preferences: sampling_params.model_preferences,
        system_prompt: sampling_params.system_prompt,
        max_tokens: sampling_params.max_tokens,
    })
}

/// MCP CreateMessageResult response type
///
/// Conforms to the MCP specification's `CreateMessageResult` interface:
///
/// ```typescript
/// interface CreateMessageResult {
///   content: TextContent | ImageContent | AudioContent;
///   role: Role;
///   model: string;
///   stopReason?: string;
///   _meta?: { [key: string]: unknown };
/// }
/// ```
///
/// Reference: [MCP specification](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-06-18/schema.mdx#createmessageresult)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingCreateMessageResult {
    pub role: String,
    pub content: McpSamplingContent,
    pub model: String,
    #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

/// Format sampling response for JSON-RPC return to MCP server
///
/// Returns a response conforming to the MCP specification's `CreateMessageResult` interface.
///
/// Reference: [MCP specification](https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-06-18/schema.mdx#createmessageresult)
pub fn format_sampling_response(response: SamplingResponse) -> Result<McpSamplingCreateMessageResult, String> {
    match response {
        SamplingResponse::Approved { llm_response, .. } => {
            Ok(McpSamplingCreateMessageResult {
                role: "assistant".to_string(),
                content: McpSamplingContent {
                    content_type: "text".to_string(),
                    text: llm_response,
                },
                model: "amazon-q".to_string(), // TODO: Use actual model name from Q CLI
                stop_reason: None,             // Phase 1: not provided
                meta: None,
            })
        },
        SamplingResponse::Rejected { reason, .. } => {
            // This case should be handled as a JSON-RPC error by the caller
            // rather than returning a CreateMessageResult
            Err(format!("User rejected sampling request: {}", reason))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sampling_request() {
        let params = serde_json::json!({
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
                "hints": [{"name": "claude-3-sonnet"}],
                "intelligencePriority": 0.8,
                "speedPriority": 0.5
            },
            "systemPrompt": "You are a helpful assistant.",
            "maxTokens": 100
        });

        let request = parse_sampling_request("test-server".to_string(), "req-123".to_string(), Some(params)).unwrap();

        assert_eq!(request.server_name, "test-server");
        assert_eq!(request.request_id, "req-123");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, "user");
        assert_eq!(request.messages[0].content.text, "What is the capital of France?");
        assert_eq!(request.system_prompt, Some("You are a helpful assistant.".to_string()));
        assert_eq!(request.max_tokens, Some(100));

        let model_prefs = request.model_preferences.unwrap();
        assert_eq!(model_prefs.intelligence_priority, Some(0.8));
        assert_eq!(model_prefs.speed_priority, Some(0.5));
    }

    #[test]
    fn test_format_sampling_response_approved() {
        let response = SamplingResponse::Approved {
            request_id: "req-123".to_string(),
            llm_response: "The capital of France is Paris.".to_string(),
        };

        let result = format_sampling_response(response).unwrap();
        assert_eq!(result.role, "assistant");
        assert_eq!(result.content.content_type, "text");
        assert_eq!(result.content.text, "The capital of France is Paris.");
        assert_eq!(result.model, "amazon-q");
        assert_eq!(result.stop_reason, None);
    }

    #[test]
    fn test_format_sampling_response_rejected() {
        let response = SamplingResponse::Rejected {
            request_id: "req-123".to_string(),
            reason: "User rejected sampling request".to_string(),
        };

        let result = format_sampling_response(response);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("User rejected sampling request"));
    }

    #[test]
    fn test_create_message_result_serialization() {
        let result = McpSamplingCreateMessageResult {
            role: "assistant".to_string(),
            content: McpSamplingContent {
                content_type: "text".to_string(),
                text: "Hello, world!".to_string(),
            },
            model: "amazon-q".to_string(),
            stop_reason: Some("max_tokens".to_string()),
            meta: None,
        };

        let json = serde_json::to_value(&result).unwrap();

        // Verify it matches MCP spec format
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"]["type"], "text");
        assert_eq!(json["content"]["text"], "Hello, world!");
        assert_eq!(json["model"], "amazon-q");
        assert_eq!(json["stopReason"], "max_tokens");

        // Verify _meta is omitted when None (due to skip_serializing_if)
        assert!(!json.as_object().unwrap().contains_key("_meta"));
    }
}
