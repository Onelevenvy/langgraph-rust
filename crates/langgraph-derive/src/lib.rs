use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Attribute, Lit, ItemFn, ReturnType};

/// Derive macro for StateGraph state types.
///
/// Annotate fields with `#[channel(reducer = "fn_name")]` to specify
/// a reducer function for that channel. Fields without the attribute
/// use LastValue (default).
///
/// # Example
///
/// ```rust,ignore
/// fn add_messages(current: &JsonValue, update: &JsonValue) -> JsonValue { /* ... */ }
///
/// #[derive(Debug, Clone, Serialize, Deserialize, Default, StateGraph)]
/// struct MyState {
///     #[channel(reducer = "add_messages")]
///     messages: Vec<Message>,
///     value: i64,  // defaults to LastValue
/// }
/// ```
#[proc_macro_derive(StateGraph, attributes(channel))]
pub fn derive_state_graph(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_state_graph(&input)
}

fn impl_state_graph(input: &DeriveInput) -> TokenStream {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("StateGraph can only be derived for structs with named fields"),
        },
        _ => panic!("StateGraph can only be derived for structs"),
    };

    let channel_registrations: Vec<proc_macro2::TokenStream> = fields
        .iter()
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            let field_name_str = field_name.to_string();

            // Check for channel attribute
            let reducer = get_channel_reducer(&field.attrs);

            match reducer {
                Some(ReducerSpec::Named(fn_name)) => {
                    let fn_ident = syn::Ident::new(&fn_name, proc_macro2::Span::call_site());
                    quote! {
                        channels.insert(
                            #field_name_str.to_string(),
                            Box::new(langgraph::channels::BinaryOperatorAggregate::new(
                                #field_name_str,
                                #fn_ident,
                            )) as Box<dyn langgraph::channels::Channel>
                        );
                    }
                }
                Some(ReducerSpec::Messages) => {
                    quote! {
                        channels.insert(
                            #field_name_str.to_string(),
                            Box::new(langgraph::channels::BinaryOperatorAggregate::new(
                                #field_name_str,
                                langgraph_prebuilt::add_messages_ref,
                            )) as Box<dyn langgraph::channels::Channel>
                        );
                    }
                }
                None => {
                    quote! {
                        channels.insert(
                            #field_name_str.to_string(),
                            Box::new(langgraph::channels::LastValue::new(#field_name_str)) as Box<dyn langgraph::channels::Channel>
                        );
                    }
                }
            }
        })
        .collect();

    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            /// Generate channels from the state struct fields.
            /// Fields with #[channel(reducer = "fn")] get BinaryOperatorAggregate,
            /// #[channel(messages)] uses the built-in add_messages reducer,
            /// others get LastValue.
            pub fn create_channels() -> std::collections::HashMap<String, Box<dyn langgraph::channels::Channel>> {
                let mut channels = std::collections::HashMap::new();
                #(#channel_registrations)*
                channels
            }
        }
    };

    TokenStream::from(expanded)
}

/// The type of reducer for a channel.
enum ReducerSpec {
    /// A named reducer function: `#[channel(reducer = "fn_name")]`
    Named(String),
    /// The built-in messages reducer: `#[channel(messages)]`
    Messages,
}

fn get_channel_reducer(attrs: &[Attribute]) -> Option<ReducerSpec> {
    for attr in attrs {
        if !attr.path().is_ident("channel") {
            continue;
        }

        let mut result = None;

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("reducer") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    result = Some(ReducerSpec::Named(s.value()));
                }
                Ok(())
            } else if meta.path.is_ident("messages") {
                result = Some(ReducerSpec::Messages);
                Ok(())
            } else {
                Err(meta.error("unknown channel attribute"))
            }
        })
        .ok();

        return result;
    }
    None
}

// ============================================================================
// #[tool] attribute macro
// ============================================================================

/// Attribute macro to define a tool from a function.
///
/// # Example
///
/// ```rust,ignore
/// use langgraph_derive::tool;
///
/// #[tool("get_weather", "Get the current weather for a location")]
/// fn get_weather(location: String) -> String {
///     format!("Weather for {}: sunny, 22°C", location)
/// }
///
/// // Usage:
/// let tool = GetWeather::new();
/// let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(tool)];
/// ```
///
/// The macro generates:
/// - A CamelCase struct (e.g., `GetWeather`)
/// - `BaseTool` trait implementation with auto-generated JSON schema
/// - `new()` and `Default` impls
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let args = parse_macro_input!(attr as ToolMacroArgs);
    impl_tool_macro(&args.name, &args.description, &func)
}

struct ToolMacroArgs {
    name: Lit,
    description: Lit,
}

impl syn::parse::Parse for ToolMacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name: Lit = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let description: Lit = input.parse()?;
        Ok(Self { name, description })
    }
}

fn impl_tool_macro(name_lit: &Lit, desc_lit: &Lit, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Get tool name and description as strings
    let tool_name = match name_lit {
        Lit::Str(s) => s.value(),
        _ => panic!("tool name must be a string literal"),
    };
    let description = match desc_lit {
        Lit::Str(s) => s.value(),
        _ => panic!("description must be a string literal"),
    };

    // Generate CamelCase struct name
    let struct_name_str = to_camel_case(&fn_name_str);
    let struct_name = syn::Ident::new(&struct_name_str, fn_name.span());

    // Extract parameters (filter out self params)
    let params: Vec<_> = func.sig.inputs.iter().filter_map(|arg| {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                return Some((pat_ident.ident.clone(), (*pat_type.ty).clone()));
            }
        }
        None
    }).collect();

    // Generate JSON schema properties
    let properties: Vec<proc_macro2::TokenStream> = params.iter().map(|(name, ty)| {
        let name_str = name.to_string();
        let json_type = rust_type_to_json_type(ty);
        quote! {
            (#name_str, serde_json::json!({"type": #json_type}))
        }
    }).collect();

    let required: Vec<String> = params.iter().map(|(name, _)| name.to_string()).collect();

    // Generate parameter extraction code
    let extractions: Vec<proc_macro2::TokenStream> = params.iter().map(|(name, ty)| {
        let name_str = name.to_string();
        let err_missing = format!("missing required parameter '{}'", name_str);
        let err_invalid = format!("invalid parameter '{}': {{}}", name_str);
        quote! {
            let #name: #ty = serde_json::from_value(
                args.get(#name_str)
                    .cloned()
                    .ok_or_else(|| langgraph_prebuilt::ToolError::InvalidArgs(#err_missing.to_string()))?
            ).map_err(|e| langgraph_prebuilt::ToolError::InvalidArgs(
                format!(#err_invalid, e)
            ))?;
        }
    }).collect();

    let param_names: Vec<_> = params.iter().map(|(name, _)| name.clone()).collect();

    // Generate the invoke body based on return type
    let is_result_return = match &func.sig.output {
        ReturnType::Type(_, ty) => {
            if let syn::Type::Path(type_path) = ty.as_ref() {
                type_path.path.segments.last()
                    .map(|s| s.ident == "Result")
                    .unwrap_or(false)
            } else {
                false
            }
        }
        _ => false,
    };

    let invoke_body = if is_result_return {
        quote! {
            #(#extractions)*
            let result = #fn_name(#(#param_names),*);
            result
                .map_err(|e| {
                    let tool_err: langgraph_prebuilt::ToolError = e.into();
                    tool_err
                })
                .and_then(|r| serde_json::to_value(r).map_err(|e| langgraph_prebuilt::ToolError::Execution(
                    format!("failed to serialize result: {}", e)
                )))
        }
    } else {
        quote! {
            #(#extractions)*
            let result = #fn_name(#(#param_names),*);
            serde_json::to_value(result).map_err(|e| langgraph_prebuilt::ToolError::Execution(
                format!("failed to serialize result: {}", e)
            ))
        }
    };

    let expanded = quote! {
        // Keep the original function
        #func

        /// Auto-generated tool struct from #[tool] macro.
        pub struct #struct_name;

        impl #struct_name {
            pub fn new() -> Self { Self }
        }

        impl Default for #struct_name {
            fn default() -> Self { Self }
        }

        #[async_trait::async_trait]
        impl langgraph_prebuilt::BaseTool for #struct_name {
            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #description
            }

            fn parameters(&self) -> Option<&serde_json::Value> {
                use std::sync::OnceLock;
                static SCHEMA: OnceLock<serde_json::Value> = OnceLock::new();
                Some(SCHEMA.get_or_init(|| {
                    let mut properties = serde_json::Map::new();
                    #(
                        {
                            let (k, v) = #properties;
                            properties.insert(k.to_string(), v);
                        }
                    )*
                    serde_json::json!({
                        "type": "object",
                        "properties": properties,
                        "required": [#(#required),*]
                    })
                }))
            }

            fn invoke(
                &self,
                args: &serde_json::Value,
                _config: &langgraph_checkpoint::config::RunnableConfig,
            ) -> Result<serde_json::Value, langgraph_prebuilt::ToolError> {
                #invoke_body
            }
        }
    };

    TokenStream::from(expanded)
}

fn to_camel_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().to_string() + &chars.as_str().to_lowercase(),
                None => String::new(),
            }
        })
        .collect()
}

fn rust_type_to_json_type(ty: &syn::Type) -> &'static str {
    if let syn::Type::Path(type_path) = ty {
        let type_name = type_path.path.segments.last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();

        match type_name.as_str() {
            "String" | "str" => "string",
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "isize" | "usize" => "integer",
            "f32" | "f64" => "number",
            "bool" => "boolean",
            _ => "string", // fallback
        }
    } else {
        "string"
    }
}
