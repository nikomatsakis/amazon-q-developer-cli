use serde_json::Value;
use std::future::Future;
use tokio::sync::mpsc;
use crate::api_client::model::ChatResponseStream;

/// Spawn a mock LLM that will follow the given script.
/// This will be running in a spawned tokio thread.
/// You can communicate with it through the returned `MockLLM` value.
pub fn spawn_mock_llm<F>(script: impl FnOnce(MockLLMContext) -> F) -> MockLLM
where
    F: Future<Output = ()> + Send + 'static,
{
    let (user_input_tx, user_input_rx) = mpsc::channel(1);
    let (llm_response_tx, llm_response_rx) = mpsc::channel(1);

    let context = MockLLMContext {
        user_input_rx,
        llm_response_tx,
    };

    tokio::spawn(script(context));

    MockLLM {
        user_input_tx,
        llm_response_rx,
    }
}

/// A mock LLM communicating with a (spawned) script.
#[derive(Debug)]
pub struct MockLLM {
    user_input_tx: mpsc::Sender<String>,
    llm_response_rx: mpsc::Receiver<ChatResponseStream>,
}

impl MockLLM {
    /// Convey the user's message to the script.
    pub async fn send_user_message(&mut self, text: String) -> Result<(), mpsc::error::SendError<String>> {
        self.user_input_tx.send(text).await
    }

    /// Read the response from the LLM (could be text or tool call).
    pub async fn read_llm_response(&mut self) -> Option<ChatResponseStream> {
        self.llm_response_rx.recv().await
    }
}

/// Mock LLM context using tokio channels for communication
pub struct MockLLMContext {
    user_input_rx: mpsc::Receiver<String>,
    llm_response_tx: mpsc::Sender<ChatResponseStream>,
}

impl MockLLMContext {
    /// Read the next user message from the channel
    pub async fn read_user_message(&mut self) -> Option<String> {
        self.user_input_rx.recv().await
    }

    /// Send a text response back to the user via channel
    pub async fn respond_to_user(&mut self, text: String) -> Result<(), mpsc::error::SendError<ChatResponseStream>> {
        self.llm_response_tx.send(ChatResponseStream::AssistantResponseEvent {
            content: text,
        }).await
    }

    /// Send a tool call via channel
    pub async fn call_tool(&mut self, tool_use_id: String, name: String, args: Option<Value>, stop: Option<bool>) -> Result<(), mpsc::error::SendError<ChatResponseStream>> {
        let input = args.map(|v| v.to_string());
        
        self.llm_response_tx.send(ChatResponseStream::ToolUseEvent {
            tool_use_id,
            name,
            input,
            stop,
        }).await
    }

    /// Mock tool invocation - just returns hardcoded responses for now
    /// This is for internal script logic, separate from call_tool which sends the tool call event
    pub async fn invoke_tool(
        &mut self,
        name: String,
        args: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        match name.as_str() {
            "countryCapital" => {
                if let Some(country) = args.get("country").and_then(|v| v.as_str()) {
                    match country {
                        "Greece" => Ok(Value::String("Athens".to_string())),
                        "France" => Ok(Value::String("Paris".to_string())),
                        _ => Ok(Value::String("Unknown".to_string())),
                    }
                } else {
                    Err("Missing country argument".into())
                }
            },
            _ => Err(format!("Unknown tool: {}", name).into()),
        }
    }
}
