//! MCP Sampling request handling
//!
//! This module contains all the functionality for handling MCP sampling requests,
//! including user approval dialogs, request editing, and response generation.

use std::sync::Arc;
use tokio::sync::Mutex;
use eyre;

use crate::mcp_client::{SamplingRequest, SamplingResponse, SamplingApproval};
use crate::util::choose;
use crate::cli::chat::{ConversationState, parser::{SendMessageStream, ResponseEvent}};
use crate::os::Os;

/// Call the LLM for a sampling request
///
/// Converts the MCP sampling request to Q CLI's conversation format,
/// calls the LLM API, and extracts the final response text.
async fn call_llm_for_sampling(
    request: &SamplingRequest,
    os: &Os,
) -> Result<String, String> {
    // Create a temporary conversation state for the sampling request
    let conversation_id = format!("sampling-{}", request.request_id);
    
    // Convert sampling request messages to MCP Prompt instances
    let prompts: std::collections::VecDeque<crate::mcp_client::Prompt> = request.messages.iter()
        .map(|msg| {
            let role = match msg.role.as_str() {
                "user" => crate::mcp_client::Role::User,
                "assistant" => crate::mcp_client::Role::Assistant,
                _ => crate::mcp_client::Role::User, // Default to user for unknown roles
            };
            
            crate::mcp_client::Prompt {
                role,
                content: crate::mcp_client::MessageContent::Text {
                    text: msg.content.text.clone(),
                },
            }
        })
        .collect();
    
    if prompts.is_empty() {
        return Err("No messages found in sampling request".to_string());
    }
    
    // Create a minimal conversation state for the sampling request
    // Note: We create a minimal state without tools/context for security
    let mut conversation_state = ConversationState::new(
        &conversation_id,
        crate::cli::agent::Agents::default(), // Empty agents for sampling
        std::collections::HashMap::new(), // No tools for sampling
        crate::cli::chat::tool_manager::ToolManager::default(), // Empty tool manager
        None, // FIXME: Use model from request.model_preferences when available
    ).await;
    
    // Use append_prompts to properly handle the MCP prompt messages
    // This ensures proper conversation history handling and role management
    let last_message = conversation_state.append_prompts(prompts);
    
    // If append_prompts returned a last message, we need to set it as the next user message
    // This is required for the conversation state to be properly prepared for LLM processing
    if let Some(last_msg_content) = last_message {
        conversation_state.set_next_user_message(last_msg_content).await;
    } else {
        return Err("No user message found in sampling request after processing prompts".to_string());
    }
    
    // Convert to sendable conversation state
    let sendable_state = conversation_state
        .as_sendable_conversation_state(os, &mut vec![], false) // No hooks for sampling
        .await
        .map_err(|e| format!("Failed to create sendable conversation state: {}", e))?;
    
    // Send the message to the LLM
    let request_metadata_lock = Arc::new(Mutex::new(None));
    let mut response_stream = SendMessageStream::send_message(
        &os.client,
        sendable_state,
        request_metadata_lock,
        None, // No message meta tags
    ).await.map_err(|e| format!("Failed to send message to LLM: {}", e))?;
    
    // Consume the response stream and extract the final text
    let mut response_text = String::new();
    
    loop {
        match response_stream.recv().await {
            Some(Ok(event)) => {
                match event {
                    ResponseEvent::AssistantText(text) => {
                        response_text.push_str(&text);
                    },
                    ResponseEvent::EndStream { .. } => {
                        break;
                    },
                    // Ignore other events (tool uses, etc.) for sampling
                    _ => {},
                }
            },
            Some(Err(e)) => {
                return Err(format!("Error receiving LLM response: {}", e));
            },
            None => {
                return Err("LLM response stream ended unexpectedly".to_string());
            },
        }
    }
    
    if response_text.trim().is_empty() {
        return Err("LLM returned empty response".to_string());
    }
    
    Ok(response_text.trim().to_string())
}

/// Handle a sampling request with trust checking and user approval
///
/// This is the main entry point for processing sampling requests. It checks if the server
/// is trusted, and either auto-approves or shows a user approval dialog accordingly.
pub async fn handle_sampling_request(
    request: &SamplingRequest,
    agent: &std::sync::Arc<tokio::sync::Mutex<crate::cli::agent::Agent>>,
    os: &Os,
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
        // FIXME: Also check trust_all_tools when we have access to the Agents struct
    };

    if is_trusted {
        // Auto-approve trusted sampling requests
        tracing::info!("Auto-approving sampling request from trusted server '{}'", server_name);
        
        match call_llm_for_sampling(request, os).await {
            Ok(llm_response) => SamplingResponse::Approved {
                request_id: request_id.clone(),
                llm_response,
            },
            Err(error) => {
                tracing::error!("LLM call failed for trusted sampling request: {}", error);
                SamplingResponse::Rejected {
                    request_id: request_id.clone(),
                    reason: format!("LLM call failed: {}", error),
                }
            }
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
                        
                        match call_llm_for_sampling(request, os).await {
                            Ok(llm_response) => SamplingResponse::Approved {
                                request_id: request_id.clone(),
                                llm_response,
                            },
                            Err(error) => {
                                tracing::error!("LLM call failed for approved sampling request: {}", error);
                                SamplingResponse::Rejected {
                                    request_id: request_id.clone(),
                                    reason: format!("LLM call failed: {}", error),
                                }
                            }
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
                        
                        match call_llm_for_sampling(request, os).await {
                            Ok(llm_response) => SamplingResponse::Approved {
                                request_id: request_id.clone(),
                                llm_response,
                            },
                            Err(error) => {
                                tracing::error!("LLM call failed for trust server sampling request: {}", error);
                                SamplingResponse::Rejected {
                                    request_id: request_id.clone(),
                                    reason: format!("LLM call failed: {}", error),
                                }
                            }
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
                format!("{}...", &msg.content.text[..97])
            } else {
                msg.content.text.clone()
            };
            format!("> {}: {}", msg.role, preview)
        } else {
            format!("{} messages", current_request.messages.len())
        };

        let prompt = format!(
            "MCP server '{}' wants to use Amazon Q:\n{}\n\nHow would you like to respond?",
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
pub fn format_sampling_request_for_editor(request: &SamplingRequest) -> String {
    let mut content = String::new();
    content.push_str(&format!("# MCP Sampling Request from '{}'\n", request.server_name));
    content.push_str("# Edit the messages below. Lines starting with # are comments and will be ignored.\n");
    content.push_str("# Format: ROLE: message content\n");
    content.push_str("# Available roles: user, assistant, system\n\n");
    
    for (i, message) in request.messages.iter().enumerate() {
        if i > 0 {
            content.push_str("\n---\n\n");
        }
        content.push_str(&format!("{}: {}\n", message.role, message.content.text));
    }
    
    content
}

/// Parse edited sampling content back into a SamplingRequest
pub fn parse_edited_sampling_content(
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
                in_message = false;
            }
            continue;
        }
        
        // Check for role: content pattern
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
        
        // If we're in a message, append to content
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
    
    // If no messages were parsed, return an error
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
    use crate::cli::agent::Agent;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[test]
    fn test_is_server_trusted_for_sampling_trusted() {
        let mut agent = Agent::default();
        
        // Trust the server for sampling
        agent.trust_server_for_sampling("file-watcher");
        
        let agent_arc = Arc::new(Mutex::new(agent));
        
        // Create a simple runtime for the async call
        let rt = tokio::runtime::Runtime::new().unwrap();
        let is_trusted = rt.block_on(async {
            let agent_guard = agent_arc.lock().await;
            agent_guard.is_server_trusted_for_sampling("file-watcher")
        });
        
        assert!(is_trusted, "Server should be trusted for sampling after being explicitly trusted");
    }

    #[test]
    fn test_is_server_trusted_for_sampling_untrusted() {
        let agent = Agent::default();
        
        let agent_arc = Arc::new(Mutex::new(agent));
        
        // Create a simple runtime for the async call
        let rt = tokio::runtime::Runtime::new().unwrap();
        let is_trusted = rt.block_on(async {
            let agent_guard = agent_arc.lock().await;
            agent_guard.is_server_trusted_for_sampling("unknown-server")
        });
        
        assert!(!is_trusted, "Unknown server should not be trusted for sampling by default");
    }

    #[test]
    fn test_is_server_trusted_for_sampling_inherited_from_tool_trust() {
        let mut agent = Agent::default();
        
        // Trust an MCP tool - should automatically trust sampling from that server
        agent.trust_mcp_tool("@docs-helper/update_readme");
        
        let agent_arc = Arc::new(Mutex::new(agent));
        
        // Create a simple runtime for the async call
        let rt = tokio::runtime::Runtime::new().unwrap();
        let is_trusted = rt.block_on(async {
            let agent_guard = agent_arc.lock().await;
            agent_guard.is_server_trusted_for_sampling("docs-helper")
        });
        
        assert!(is_trusted, "Server should be trusted for sampling when its tools are trusted");
    }
}
