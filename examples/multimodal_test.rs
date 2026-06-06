use dotenvy::dotenv;
use langgraph_prebuilt::{BaseChatModel, Message, MessageContent, ContentBlock, ImageUrl};
use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY not set");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "mimo-v2.5".to_string());
    
    println!("API Base: {:?}", api_base);
    println!("Model: {}", model_name);
    
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        ..Default::default()
    });
    
    // Construct message with text and image
    let messages = vec![
        Message::Human {
            content: MessageContent::Blocks(vec![
                ContentBlock::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example-files.cnbj1.mi-fds.com/example-files/image/image_example.png".to_string(),
                        detail: None,
                    }
                },
                ContentBlock::Text {
                    text: "please describe the content of the image".to_string(),
                }
            ]),
            id: None,
        }
    ];
    
    println!("Calling LLM with multimodal message...");
    let response = model.ainvoke(&messages, &Default::default()).await?;
    
    println!("Response:");
    println!("{}", response);
    
    Ok(())
}
