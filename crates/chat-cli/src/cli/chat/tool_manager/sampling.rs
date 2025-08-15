//! MCP Sampling request handling
//!
//! This module contains all the functionality for handling MCP sampling requests,
//! including user approval dialogs, request editing, and response generation.

use std::collections::VecDeque;
use std::sync::Arc;

use eyre;
use regex::Regex;
use tokio::sync::Mutex;

use crate::cli::chat::ConversationState;
use crate::cli::chat::parser::{ResponseEvent, SendMessageStream};
use crate::mcp_client::{
    McpSamplingMessage, MessageContent, Prompt, Role, SamplingApproval, SamplingRequest, SamplingResponse,
};
use crate::os::Os;
use crate::util::choose;

/// Call the LLM for a sampling request
///
/// Converts the MCP sampling request to Q CLI's conversation format,
/// calls the LLM API, and extracts the final response text.
async fn call_llm_for_sampling(request: &SamplingRequest, os: &Os) -> Result<String, String> {
    // Create a temporary conversation state for the sampling request
    let conversation_id = format!("sampling-{}", request.request_id);

    if request.messages.is_empty() {
        return Err("No messages found in sampling request".to_string());
    }

    // Convert sampling request messages to MCP Prompt instances
    let prompts = sampling_messages_to_prompt(&request.messages)?;

    // Create a minimal conversation state for the sampling request
    // Note: We create a minimal state without tools/context for security
    let mut conversation_state = ConversationState::new(
        &conversation_id,
        crate::cli::agent::Agents::default(), // Empty agents for sampling
        std::collections::HashMap::new(),     // No tools for sampling
        crate::cli::chat::tool_manager::ToolManager::default(), // Empty tool manager
        None,                                 // FIXME: Use model from request.model_preferences when available
        os,
    )
    .await;

    // Use append_prompts to properly handle the MCP prompt messages
    // This ensures proper conversation history handling and role management
    let Some(last_message_content) = conversation_state.append_prompts(prompts) else {
        panic!("`sampling_messages_to_prompt` ensures final message is a user message")
    };
    conversation_state.set_next_user_message(last_message_content).await;

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
    )
    .await
    .map_err(|e| format!("Failed to send message to LLM: {}", e))?;

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

/// Convert a series of sampling messages into a set of prompts suitable for appending to a
/// conversation. For reasons I don't entirely know, `ConversationState::append_prompts` expects
/// user/assistant messages to come in pairs, and we enforce that invariant, concatenating
/// consequence user/assisstant messages together. Returns `Err` if the final message is not a user
/// message.
fn sampling_messages_to_prompt(msgs: &[McpSamplingMessage]) -> Result<VecDeque<Prompt>, String> {
    // First pass, create pairs of roles and strings
    let mut pairs: Vec<(Role, String)> = Vec::new();
    for msg in msgs {
        // Extract text from MessageContent
        let text = match &msg.content {
            MessageContent::Text { text } => text.clone(),
            MessageContent::Image { .. } => {
                return Err("Image content not supported in sampling messages".to_string());
            },
            MessageContent::Resource { .. } => {
                return Err("Resource content not supported in sampling messages".to_string());
            },
        };

        // Append consecutive messages as paragraphs.
        if let Some((last_role, last_text)) = pairs.last_mut() {
            if *last_role == msg.role {
                last_text.push_str("\n\n");
                last_text.push_str(&text);
                continue;
            }
        }

        pairs.push((msg.role.clone(), text));
    }

    // Make sure that the last message is a user message, adding a dummy message if needed.
    let last_pair_is_user = pairs.last().map(|(role, _)| *role == Role::User).unwrap_or(false);
    if !last_pair_is_user {
        return Err("Final message is not a user message".to_string());
    }

    // Second pass, convert to proper prompts
    Ok(pairs
        .into_iter()
        .map(|(role, text)| Prompt {
            role,
            content: MessageContent::Text { text },
        })
        .collect())
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
            },
        }
    } else {
        // Show user approval dialog for untrusted requests
        match show_sampling_approval_dialog(request) {
            Ok(approval) => {
                match approval {
                    SamplingApproval::Approve(_approved_request) => {
                        tracing::info!("User approved sampling request from server '{}'", server_name);

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
                            },
                        }
                    },
                    SamplingApproval::TrustServer => {
                        tracing::info!("User chose to trust sampling from server '{}'", server_name);
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
                            },
                        }
                    },
                    SamplingApproval::Reject => {
                        tracing::info!("User rejected sampling request from server '{}'", server_name);
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
fn show_sampling_approval_dialog(request: &SamplingRequest) -> eyre::Result<SamplingApproval> {
    let mut current_request = request.clone();

    loop {
        // Format the current sampling request for display
        let messages_preview = if current_request.messages.len() == 1 {
            let msg = &current_request.messages[0];
            let preview = match &msg.content {
                MessageContent::Text { text } => {
                    if text.len() > 100 {
                        format!("{}...", &text[..97])
                    } else {
                        text.clone()
                    }
                },
                MessageContent::Image { .. } => "[Image content]".to_string(),
                MessageContent::Resource { .. } => "[Resource content]".to_string(),
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
                    },
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
fn open_editor_for_sampling(request: &SamplingRequest) -> eyre::Result<SamplingRequest> {
    // Format the sampling request as editable text
    let formatted_content = format_sampling_request_for_editor(request)?;

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
pub fn format_sampling_request_for_editor(request: &SamplingRequest) -> eyre::Result<String> {
    use std::fmt::Write;

    let mut content = String::new();

    writeln!(content, "# MCP Sampling Request from '{}'", request.server_name).unwrap();
    writeln!(content, "#").unwrap();
    writeln!(content, "# Edit the messages below.").unwrap();
    writeln!(content, "#").unwrap();
    writeln!(content, "# Lines starting with comments are ignored.").unwrap();
    writeln!(content, "#").unwrap();
    writeln!(content, "# Each message is formatted like").unwrap();
    writeln!(content, "#").unwrap();
    writeln!(content, "# ```$role").unwrap();
    writeln!(content, "# ...").unwrap();
    writeln!(content, "# ```").unwrap();
    writeln!(content, "# where `$role` is either `user` or `assistant`.").unwrap();
    writeln!(content, "").unwrap();

    for message in &request.messages {
        let text = match &message.content {
            MessageContent::Text { text } => text,
            MessageContent::Image { .. } => eyre::bail!("cannot edit message with image content"),
            MessageContent::Resource { .. } => eyre::bail!("cannot edit message with resource content"),
        };
        let delimiter = delimiter(text);

        // add an extra newline at the end of the text if there isn't already one
        let end_nl = if text.ends_with("\n") { "" } else { "\n" };

        write!(
            content,
            "{delimiter}{role}\n{text}{end_nl}{delimiter}\n\n",
            role = message.role
        )
        .unwrap();
    }

    Ok(content)
}

/// Create a delimiter consisting of at least three ticks
fn delimiter(s: &str) -> String {
    let mut d = String::from("```");
    while s.contains(&d) {
        d.push('`');
    }
    d
}

/// Parse edited sampling content back into a SamplingRequest
///
/// Expected format is described in `{format_sampling_request_for_editor}`.
pub fn parse_edited_sampling_content(
    content: &str,
    original_request: &SamplingRequest,
) -> eyre::Result<SamplingRequest> {
    use crate::mcp_client::McpSamplingMessage;

    let regex = Regex::new(r"(?P<delim>```+)(?P<role>[a-z]+)").unwrap();

    let mut messages = Vec::new();
    let mut iterator = content.lines().zip(1..);
    'outer: while let Some((line, index)) = iterator.next() {
        let line = line.trim();

        // Skip comments and empty lines
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        // Expect an open verbatim section.
        let Some(m) = regex.captures(line) else {
            eyre::bail!("on line {index}, expected \"```$role\" or something similar");
        };
        let delimiter = &m["delim"];
        let role = match &m["role"] {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            r => eyre::bail!("on line {index}, unrecognied role `{r}`, expected `user` or `assistant`"),
        };

        // Scan until we find the delimiter.
        let mut accumulated = String::new();
        while let Some((content, _)) = iterator.next() {
            if content == delimiter {
                messages.push(McpSamplingMessage {
                    role,
                    content: MessageContent::Text { text: accumulated },
                });
                continue 'outer;
            }
            accumulated.push_str(content);
            accumulated.push('\n');
        }

        eyre::bail!("on line {index}, closing delimiter not found")
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
    use super::{delimiter, format_sampling_request_for_editor, parse_edited_sampling_content};
    use crate::cli::agent::Agent;
    use crate::cli::chat::tool_manager::sampling::sampling_messages_to_prompt;
    use crate::mcp_client::{McpSamplingMessage, MessageContent, Prompt, Role, SamplingRequest};

    macro_rules! assert_json_eq {
        ($left:expr, $right:expr, $($msg:tt)*) => {
            assert_eq!(
                serde_json::to_string_pretty(&$left).unwrap(),
                serde_json::to_string_pretty(&$right).unwrap(),
                $($msg)*
            )
        };
    }

    // Helper function to create a user message for testing
    fn create_user_message(text: &str) -> McpSamplingMessage {
        McpSamplingMessage {
            role: Role::User,
            content: MessageContent::Text { text: text.to_string() },
        }
    }

    // Helper function to create an assistant message for testing
    fn create_assistant_message(text: &str) -> McpSamplingMessage {
        McpSamplingMessage {
            role: Role::Assistant,
            content: MessageContent::Text { text: text.to_string() },
        }
    }

    #[test]
    fn test_is_server_trusted_for_sampling_trusted() {
        let mut agent = Agent::default();

        assert!(
            !agent.is_server_trusted_for_sampling("file-watcher"),
            "Unknown server should not be trusted for sampling by default"
        );
        assert!(
            !agent.is_server_trusted_for_sampling("temperature-sensor"),
            "Unknown server should not be trusted for sampling by default"
        );

        // Trust the server for sampling
        agent.trust_server_for_sampling("file-watcher");

        assert!(
            agent.is_server_trusted_for_sampling("file-watcher"),
            "Server should be trusted for sampling after being explicitly trusted"
        );
        assert!(
            !agent.is_server_trusted_for_sampling("temperature-sensor"),
            "Unknown server should not be trusted for sampling by default"
        );
    }

    #[test]
    fn test_is_server_trusted_for_sampling_no_inheritance_from_tool_trust() {
        let mut agent = Agent::default();

        // Trust an MCP tool - should NOT automatically trust sampling from that server
        agent.trust_mcp_tool("@docs-helper/update_readme");

        assert!(
            !agent.is_server_trusted_for_sampling("docs-helper"),
            "Server should NOT be trusted for sampling when only its tools are trusted (conservative approach)"
        );

        // But explicit sampling trust should still work
        agent.trust_server_for_sampling("docs-helper");
        assert!(
            agent.is_server_trusted_for_sampling("docs-helper"),
            "Server should be trusted for sampling when explicitly trusted"
        );
    }

    #[test]
    fn test_convert_sampling_messages_to_prompts_multiple_assistant() {
        let messages = vec![
            create_assistant_message("You are an indifferent agent."),
            create_assistant_message("You help only when sufficiently bored."),
            create_user_message("Hi!"),
        ];
        let prompts = sampling_messages_to_prompt(&messages).unwrap();
        assert_eq!(prompts.len(), 2, "Should convert single message to single prompt");

        assert_json_eq!(
            prompts[0],
            Prompt {
                role: Role::Assistant,
                content: MessageContent::Text {
                    text: format!("You are an indifferent agent.\n\nYou help only when sufficiently bored.")
                }
            },
            "Should concatenate consecutive assistant messages"
        );

        assert_json_eq!(
            prompts[1],
            Prompt {
                role: Role::User,
                content: MessageContent::Text { text: format!("Hi!") }
            },
            "Should end with user"
        );
    }

    #[test]
    fn test_convert_sampling_messages_to_prompts_single_user() {
        let messages = vec![create_user_message("What is the capital of France?")];
        let prompts = sampling_messages_to_prompt(&messages).unwrap();
        assert_json_eq!(
            prompts[0],
            Prompt {
                role: Role::User,
                content: MessageContent::Text {
                    text: format!("What is the capital of France?")
                }
            },
            "Should end with user"
        );
    }

    #[test]
    fn test_convert_sampling_messages_to_prompts_conversation() {
        let messages = vec![
            create_user_message("What is the capital of France?"),
            create_assistant_message("The capital of France is Paris."),
            create_user_message("What about Germany?"),
        ];
        let prompts = sampling_messages_to_prompt(&messages).unwrap();

        assert_eq!(prompts.len(), 3);

        assert_json_eq!(
            prompts[0],
            Prompt {
                role: Role::User,
                content: MessageContent::Text {
                    text: format!("What is the capital of France?")
                }
            },
            "Should end with user"
        );

        assert_json_eq!(
            prompts[1],
            Prompt {
                role: Role::Assistant,
                content: MessageContent::Text {
                    text: format!("The capital of France is Paris.")
                }
            },
            "Should end with user"
        );

        assert_json_eq!(
            prompts[2],
            Prompt {
                role: Role::User,
                content: MessageContent::Text {
                    text: format!("What about Germany?")
                }
            },
            "Should end with user"
        );
    }

    #[test]
    fn test_unknown_role_errors() {
        // Since we now use Role enum, we can't create messages with unknown roles
        // This test now verifies that only User and Assistant roles are supported
        let user_msg = create_user_message("User message");
        let assistant_msg = create_assistant_message("Assistant message");

        assert!(matches!(user_msg.role, Role::User));
        assert!(matches!(assistant_msg.role, Role::Assistant));

        // Test that sampling_messages_to_prompt works with valid roles
        let messages = vec![user_msg];
        let result = sampling_messages_to_prompt(&messages);
        assert!(result.is_ok());
    }

    #[test]
    fn test_no_user_errors() {
        let messages = vec![McpSamplingMessage {
            role: Role::Assistant,
            content: MessageContent::Text {
                text: "You are a helpful assistant.".to_string(),
            },
        }];

        sampling_messages_to_prompt(&messages).unwrap_err();
    }

    // Helper function to create a test SamplingRequest
    fn create_test_sampling_request() -> SamplingRequest {
        SamplingRequest {
            server_name: "test-server".to_string(),
            request_id: "test-123".to_string(),
            messages: vec![
                create_user_message("Hello, how are you?"),
                create_assistant_message("I'm doing well, thank you!"),
                create_user_message("What's the weather like?"),
            ],
            model_preferences: None,
            system_prompt: None,
            max_tokens: Some(100),
        }
    }

    #[test]
    fn test_format_sampling_request_for_editor() {
        let request = create_test_sampling_request();
        let formatted = format_sampling_request_for_editor(&request).unwrap();

        // Check that it contains the header
        assert!(formatted.contains("# MCP Sampling Request from 'test-server'"));
        assert!(formatted.contains("# Edit the messages below."));
        assert!(formatted.contains("# Lines starting with comments are ignored."));
        assert!(formatted.contains("# Each message is formatted like"));
        assert!(formatted.contains("# ```$role"));

        // Check that it contains the messages in the correct format
        assert!(formatted.contains("```user\nHello, how are you?\n```"));
        assert!(formatted.contains("```assistant\nI'm doing well, thank you!\n```"));
        assert!(formatted.contains("```user\nWhat's the weather like?\n```"));
    }

    #[test]
    fn test_format_sampling_request_for_editor_with_backticks_in_content() {
        let request = SamplingRequest {
            server_name: "test-server".to_string(),
            request_id: "test-123".to_string(),
            messages: vec![
                create_user_message("Here's some code: ```rust\nfn main() {}\n```"),
                create_assistant_message("That's a simple ```main``` function."),
            ],
            model_preferences: None,
            system_prompt: None,
            max_tokens: Some(100),
        };

        let formatted = format_sampling_request_for_editor(&request).unwrap();

        // Should use longer delimiters when content contains backticks
        assert!(formatted.contains("````user\nHere's some code: ```rust\nfn main() {}\n```\n````"));
        assert!(formatted.contains("````assistant\nThat's a simple ```main``` function.\n````"));
    }

    #[test]
    fn test_parse_edited_sampling_content_basic() {
        let original_request = create_test_sampling_request();
        let content = r#"# MCP Sampling Request from 'test-server'
#
# Edit the messages below.

```user
Hello there!
```

```assistant
Hi! How can I help you today?
```

```user
Tell me a joke.
```
"#;

        let parsed = parse_edited_sampling_content(content, &original_request).unwrap();

        assert_eq!(parsed.server_name, "test-server");
        assert_eq!(parsed.request_id, "test-123");
        assert_eq!(parsed.messages.len(), 3);

        assert_eq!(parsed.messages[0].role, Role::User);
        if let MessageContent::Text { text } = &parsed.messages[0].content {
            assert_eq!(text, "Hello there!\n");
        } else {
            panic!("Expected text content");
        }

        assert_eq!(parsed.messages[1].role, Role::Assistant);
        if let MessageContent::Text { text } = &parsed.messages[1].content {
            assert_eq!(text, "Hi! How can I help you today?\n");
        } else {
            panic!("Expected text content");
        }

        assert_eq!(parsed.messages[2].role, Role::User);
        if let MessageContent::Text { text } = &parsed.messages[2].content {
            assert_eq!(text, "Tell me a joke.\n");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_parse_edited_sampling_content_with_longer_delimiters() {
        let original_request = create_test_sampling_request();
        let content = r#"````user
Here's some code: ```rust
fn main() {}
```
````

````assistant
That's a simple ```main``` function.
````
"#;

        let parsed = parse_edited_sampling_content(content, &original_request).unwrap();

        assert_eq!(parsed.messages.len(), 2);

        if let MessageContent::Text { text } = &parsed.messages[0].content {
            assert_eq!(text, "Here's some code: ```rust\nfn main() {}\n```\n");
        } else {
            panic!("Expected text content");
        }

        if let MessageContent::Text { text } = &parsed.messages[1].content {
            assert_eq!(text, "That's a simple ```main``` function.\n");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_parse_edited_sampling_content_skips_comments_and_empty_lines() {
        let original_request = create_test_sampling_request();
        let content = r#"# This is a comment
# Another comment

```user
Hello!
```

# More comments here

```assistant
Hi there!
```

# Final comment
"#;

        let parsed = parse_edited_sampling_content(content, &original_request).unwrap();

        assert_eq!(parsed.messages.len(), 2);
        assert_eq!(parsed.messages[0].role, Role::User);
        assert_eq!(parsed.messages[1].role, Role::Assistant);
    }

    #[test]
    fn test_parse_edited_sampling_content_invalid_role() {
        let original_request = create_test_sampling_request();
        let content = r#"```system
You are a helpful assistant.
```"#;

        let result = parse_edited_sampling_content(content, &original_request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unrecognied role `system`"));
    }

    #[test]
    fn test_parse_edited_sampling_content_missing_closing_delimiter() {
        let original_request = create_test_sampling_request();
        let content = r#"```user
Hello there!
"#;

        let result = parse_edited_sampling_content(content, &original_request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("closing delimiter not found"));
    }

    #[test]
    fn test_parse_edited_sampling_content_no_messages() {
        let original_request = create_test_sampling_request();
        let content = r#"# Just comments
# No actual messages
"#;

        let result = parse_edited_sampling_content(content, &original_request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No valid messages found"));
    }

    #[test]
    fn test_parse_edited_sampling_content_invalid_format() {
        let original_request = create_test_sampling_request();
        let content = r#"This is not a valid format
No delimiters here
"#;

        let result = parse_edited_sampling_content(content, &original_request);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected \"```$role\""));
    }

    #[test]
    fn test_delimiter_function() {
        // Test the delimiter function with various inputs
        assert_eq!(delimiter("simple text"), "```");
        assert_eq!(delimiter("text with ``` backticks"), "````");
        assert_eq!(delimiter("text with ```` four backticks"), "`````");
        assert_eq!(delimiter("no backticks here"), "```");
    }

    #[test]
    fn test_round_trip_format_and_parse() {
        let original_request = create_test_sampling_request();

        // Format the request
        let formatted = format_sampling_request_for_editor(&original_request).unwrap();

        // Parse it back
        let parsed = parse_edited_sampling_content(&formatted, &original_request).unwrap();

        // Should have the same messages (content-wise)
        assert_eq!(parsed.messages.len(), original_request.messages.len());

        for (original, parsed) in original_request.messages.iter().zip(parsed.messages.iter()) {
            assert_eq!(original.role, parsed.role);

            if let (MessageContent::Text { text: orig_text }, MessageContent::Text { text: parsed_text }) =
                (&original.content, &parsed.content)
            {
                // The parsed version will have a trailing newline if the original didn't
                let expected_text = if orig_text.ends_with('\n') {
                    orig_text.clone()
                } else {
                    format!("{}\n", orig_text)
                };
                assert_eq!(expected_text, *parsed_text);
            } else {
                panic!("Expected text content in both messages");
            }
        }

        // Metadata should be preserved
        assert_eq!(parsed.server_name, original_request.server_name);
        assert_eq!(parsed.request_id, original_request.request_id);
        assert_eq!(parsed.max_tokens, original_request.max_tokens);
    }
}
