//! MCP Sampling request handling
//!
//! This module contains all the functionality for handling MCP sampling requests,
//! including user approval dialogs, request editing, and response generation.

use crate::mcp_client::{SamplingRequest, SamplingResponse, SamplingApproval};
use crate::util::choose;

/// Handle a sampling request with trust checking and user approval
///
/// This is the main entry point for processing sampling requests. It checks if the server
/// is trusted, and either auto-approves or shows a user approval dialog accordingly.
pub async fn handle_sampling_request(
    request: &SamplingRequest,
    agent: &std::sync::Arc<tokio::sync::Mutex<crate::cli::agent::Agent>>,
) -> SamplingResponse {
    let server_name = &request.server_name;
    let request_id = &request.request_id;

    tracing::info!(
        "Received sampling request from server '{}' with request_id '{}'",
        server_name,
        request_id
    );

    // Check if sampling is trusted for this server
    let is_trusted = {
        let agent_guard = agent.lock().await;
        agent_guard.is_server_trusted_for_sampling(server_name)
        // TODO: Also check trust_all_tools when we have access to the Agents struct
    };

    if is_trusted {
        // Auto-approve trusted sampling requests
        tracing::info!("Auto-approving sampling request from trusted server '{}'", server_name);
        // TODO: Make actual LLM call here
        SamplingResponse::Approved {
            request_id: request_id.clone(),
            llm_response: "This is a placeholder response for trusted sampling request.".to_string(),
        }
    } else {
        // Show user approval dialog for untrusted requests
        match show_sampling_approval_dialog(request) {
            Ok(approval) => {
                match approval {
                    SamplingApproval::Approve(_approved_request) => {
                        tracing::info!(
                            "User approved sampling request from server '{}'",
                            server_name
                        );
                        // TODO: Make actual LLM call with approved_request here
                        SamplingResponse::Approved {
                            request_id: request_id.clone(),
                            llm_response: "This is a placeholder response for approved sampling request.".to_string(),
                        }
                    },
                    SamplingApproval::TrustServer => {
                        tracing::info!(
                            "User chose to trust sampling from server '{}'",
                            server_name
                        );
                        // Add the server to trusted sampling
                        {
                            let mut agent_guard = agent.lock().await;
                            agent_guard.trust_server_for_sampling(server_name);
                        }
                        // TODO: Make actual LLM call here
                        SamplingResponse::Approved {
                            request_id: request_id.clone(),
                            llm_response: "This is a placeholder response for trusted sampling request.".to_string(),
                        }
                    },
                    SamplingApproval::Reject => {
                        tracing::info!(
                            "User rejected sampling request from server '{}'",
                            server_name
                        );
                        SamplingResponse::Rejected {
                            request_id: request_id.clone(),
                            reason: "User rejected the sampling request".to_string(),
                        }
                    },
                }
            },
            Err(e) => {
                tracing::error!("Error showing sampling approval dialog: {}", e);
                SamplingResponse::Rejected {
                    request_id: request_id.clone(),
                    reason: format!("Error in approval dialog: {}", e),
                }
            },
        }
    }
}

/// Show user approval dialog for sampling requests
///
/// Displays a dialog asking the user whether to approve, reject, trust, or edit
/// sampling requests from the given MCP server. The edit option allows users to
/// modify the request content and then returns to the approval dialog.
fn show_sampling_approval_dialog(
    request: &SamplingRequest,
) -> eyre::Result<SamplingApproval> {
    let mut current_request = request.clone();
    
    loop {
        // Format the current sampling request for display
        let messages_preview = if current_request.messages.len() == 1 {
            let msg = &current_request.messages[0];
            let preview = if msg.content.text.len() > 100 {
                format!("{}...", &msg.content.text[..100])
            } else {
                msg.content.text.clone()
            };
            format!("{}: {}", msg.role, preview)
        } else {
            format!("{} messages in conversation", current_request.messages.len())
        };

        let prompt = format!(
            "MCP Sampling Request from '{}'\n\n{}\n\nHow would you like to respond?",
            current_request.server_name, messages_preview
        );

        let options = vec![
            "Approve once",
            "Edit in editor",
            "Reject",
            "Trust this server (approve all future sampling)",
        ];

        match choose(&prompt, &options)? {
            Some(0) => {
                // Return the current request (which may or may not have been edited)
                return Ok(SamplingApproval::Approve(current_request));
            },
            Some(1) => {
                // Open editor with the current sampling request content
                match open_editor_for_sampling(&current_request) {
                    Ok(edited_request) => {
                        current_request = edited_request;
                        // Continue the loop to show the dialog again with edited content
                        continue;
                    },
                    Err(e) => {
                        tracing::error!("Failed to open editor for sampling request: {}", e);
                        // Continue the loop to let user try again or choose different option
                        continue;
                    }
                }
            },
            Some(2) => return Ok(SamplingApproval::Reject),
            Some(3) => return Ok(SamplingApproval::TrustServer),
            Some(_) => unreachable!("Invalid selection index"),
            None => {
                // User cancelled (Ctrl+C)
                tracing::info!("User cancelled sampling approval dialog");
                return Ok(SamplingApproval::Reject);
            },
        }
    }
}

/// Open editor for sampling request editing
///
/// Formats the sampling request as human-readable text, opens it in the user's
/// preferred editor, and parses the result back into a SamplingRequest.
fn open_editor_for_sampling(
    request: &SamplingRequest,
) -> eyre::Result<SamplingRequest> {
    // Format the sampling request as editable text
    let formatted_content = format_sampling_request_for_editor(request);
    
    // Use the existing editor functionality
    match crate::cli::chat::cli::editor::open_editor(Some(formatted_content)) {
        Ok(edited_content) => {
            // Parse the edited content back into a SamplingRequest
            parse_edited_sampling_content(&edited_content, request)
        },
        Err(e) => Err(eyre::eyre!("Editor failed: {}", e)),
    }
}

/// Format sampling request for editor display
fn format_sampling_request_for_editor(request: &SamplingRequest) -> String {
    let mut content = String::new();
    content.push_str(&format!("# MCP Sampling Request from '{}'\n", request.server_name));
    content.push_str("# Edit the messages below. Lines starting with # are comments and will be ignored.\n");
    content.push_str("# Format: ROLE: message content\n");
    content.push_str("# Available roles: user, assistant, system\n\n");
    
    for (i, message) in request.messages.iter().enumerate() {
        if i > 0 {
            content.push_str("---\n");
        }
        content.push_str(&format!("{}: {}\n", message.role, message.content.text));
    }
    
    content
}

/// Parse edited sampling content back into a SamplingRequest
fn parse_edited_sampling_content(
    content: &str,
    original_request: &SamplingRequest,
) -> eyre::Result<SamplingRequest> {
    use crate::mcp_client::{McpSamplingMessage, McpSamplingContent};
    
    let mut messages = Vec::new();
    let mut current_role = String::new();
    let mut current_content = String::new();
    let mut in_message = false;
    
    for line in content.lines() {
        let line = line.trim();
        
        // Skip comments and empty lines
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        
        // Check for separator
        if line == "---" {
            // Finish current message if we have one
            if in_message && !current_role.is_empty() {
                messages.push(McpSamplingMessage {
                    role: current_role.clone(),
                    content: McpSamplingContent {
                        content_type: "text".to_string(),
                        text: current_content.trim().to_string(),
                    },
                });
                current_role.clear();
                current_content.clear();
            }
            in_message = false;
            continue;
        }
        
        // Check if this line starts a new message (role: content format)
        if let Some(colon_pos) = line.find(':') {
            let potential_role = line[..colon_pos].trim().to_lowercase();
            if matches!(potential_role.as_str(), "user" | "assistant" | "system") {
                // Finish previous message if we have one
                if in_message && !current_role.is_empty() {
                    messages.push(McpSamplingMessage {
                        role: current_role.clone(),
                        content: McpSamplingContent {
                            content_type: "text".to_string(),
                            text: current_content.trim().to_string(),
                        },
                    });
                }
                
                // Start new message
                current_role = potential_role;
                current_content = line[colon_pos + 1..].trim().to_string();
                in_message = true;
                continue;
            }
        }
        
        // If we're in a message, add this line to the content
        if in_message {
            if !current_content.is_empty() {
                current_content.push('\n');
            }
            current_content.push_str(line);
        }
    }
    
    // Finish the last message
    if in_message && !current_role.is_empty() {
        messages.push(McpSamplingMessage {
            role: current_role,
            content: McpSamplingContent {
                content_type: "text".to_string(),
                text: current_content.trim().to_string(),
            },
        });
    }
    
    if messages.is_empty() {
        return Err(eyre::eyre!("No valid messages found in edited content"));
    }
    
    // Create new request with edited messages
    Ok(SamplingRequest {
        server_name: original_request.server_name.clone(),
        request_id: original_request.request_id.clone(),
        messages,
        model_preferences: original_request.model_preferences.clone(),
        system_prompt: original_request.system_prompt.clone(),
        max_tokens: original_request.max_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_client::{McpSamplingContent, McpSamplingMessage, SamplingRequest};

    #[test]
    fn test_sampling_approval_dialog_formatting() {
        // Test single message formatting
        let request = SamplingRequest {
            server_name: "test-server".to_string(),
            request_id: "req-123".to_string(),
            messages: vec![McpSamplingMessage {
                role: "user".to_string(),
                content: McpSamplingContent {
                    content_type: "text".to_string(),
                    text: "What is the capital of France?".to_string(),
                },
            }],
            model_preferences: None,
            system_prompt: None,
            max_tokens: None,
        };

        // Just test that the function doesn't panic - we can't easily test the UI
        let formatted = format_sampling_request_for_editor(&request);
        assert!(formatted.contains("test-server"));
        assert!(formatted.contains("What is the capital of France?"));
    }

    #[test]
    fn test_parse_edited_sampling_content_single_message() {
        let original_request = SamplingRequest {
            server_name: "test-server".to_string(),
            request_id: "req-123".to_string(),
            messages: vec![],
            model_preferences: None,
            system_prompt: None,
            max_tokens: None,
        };

        let edited_content = r#"
# MCP Sampling Request from 'test-server'
# Edit the messages below. Lines starting with # are comments.

user: What is the capital of France?
"#;

        let result = parse_edited_sampling_content(edited_content, &original_request).unwrap();
        
        assert_eq!(result.server_name, "test-server");
        assert_eq!(result.request_id, "req-123");
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[0].content.text, "What is the capital of France?");
    }

    #[test]
    fn test_parse_edited_sampling_content_multi_message() {
        let original_request = SamplingRequest {
            server_name: "test-server".to_string(),
            request_id: "req-456".to_string(),
            messages: vec![],
            model_preferences: None,
            system_prompt: None,
            max_tokens: None,
        };

        let edited_content = r#"
user: What is the capital of France?
---
assistant: The capital of France is Paris.
---
user: What about Germany?
"#;

        let result = parse_edited_sampling_content(edited_content, &original_request).unwrap();
        
        assert_eq!(result.messages.len(), 3);
        
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[0].content.text, "What is the capital of France?");
        
        assert_eq!(result.messages[1].role, "assistant");
        assert_eq!(result.messages[1].content.text, "The capital of France is Paris.");
        
        assert_eq!(result.messages[2].role, "user");
        assert_eq!(result.messages[2].content.text, "What about Germany?");
    }

    #[test]
    fn test_format_sampling_request_for_editor() {
        let request = SamplingRequest {
            server_name: "test-server".to_string(),
            request_id: "req-format".to_string(),
            messages: vec![
                McpSamplingMessage {
                    role: "user".to_string(),
                    content: McpSamplingContent {
                        content_type: "text".to_string(),
                        text: "What is the capital of France?".to_string(),
                    },
                },
                McpSamplingMessage {
                    role: "assistant".to_string(),
                    content: McpSamplingContent {
                        content_type: "text".to_string(),
                        text: "The capital of France is Paris.".to_string(),
                    },
                },
            ],
            model_preferences: None,
            system_prompt: None,
            max_tokens: None,
        };

        let formatted = format_sampling_request_for_editor(&request);
        
        assert!(formatted.contains("# MCP Sampling Request from 'test-server'"));
        assert!(formatted.contains("# Available roles: user, assistant, system"));
        assert!(formatted.contains("user: What is the capital of France?"));
        assert!(formatted.contains("---"));
        assert!(formatted.contains("assistant: The capital of France is Paris."));
    }
}
