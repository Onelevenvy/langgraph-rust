use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Attribute, Lit, ItemFn, ReturnType};

/// Derive macro for StateGraph state types.
///
/// Annotate fields with `#[channel(reducer = "fn_name")]` to specify
/// a reducer function for that channel. Fields without the attribute
/// use LastValue (default).
///
/// **Robustness Check**: This macro enforces that every field must have 
/// `#[serde(default)]` (or be an `Option` which handles missing keys gracefully).
/// This prevents silent state loss during graph resume operations.
#[proc_macro_derive(StateGraph, attributes(channel))]
pub fn derive_state_graph(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_state_graph(&input)
}


/// This attribute macro:
/// 1. Automatically adds `#[derive(serde::Serialize, serde::Deserialize, Clone, Default, StateGraph)]`.
/// 2. Automatically injects `#[serde(default)]` on every field to ensure robustness.
/// 
/// Usage:
/// ```rust,ignore
/// #[langgraph_state]
/// struct MyState {
///     #[channel(messages)]
///     messages: Vec<Message>,
///     other_field: String,
/// }
/// ```
#[proc_macro_attribute]
pub fn langgraph_state(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as syn::ItemStruct);
    
    // 1. Add the "big bunch" of derives
    input.attrs.push(syn::parse_quote! {
        #[derive(serde::Serialize, serde::Deserialize, Clone, Default, langgraph_derive::StateGraph)]
    });

    // 2. Walk fields and ensure #[serde(default)] exists
    if let syn::Fields::Named(fields) = &mut input.fields {
        for field in &mut fields.named {
            let mut has_default = false;
            for attr in &field.attrs {
                if attr.path().is_ident("serde") {
                    let _ = attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident("default") {
                            has_default = true;
                        }
                        Ok(())
                    });
                }
            }

            if !has_default {
                field.attrs.push(syn::parse_quote! {
                    #[serde(default)]
                });
            }
        }
    }

    let expanded = quote! {
        #input
    };

    TokenStream::from(expanded)
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

    // ── ROBUSTNESS CHECK ─────────────────────────────────────────────────────
    // For every field, ensure it has #[serde(default)]
    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        
        let mut has_serde_default = false;
        for attr in &field.attrs {
            if attr.path().is_ident("serde") {
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("default") {
                        has_serde_default = true;
                    }
                    Ok(())
                });
            }
        }

        if !has_serde_default {
            let error_msg = format!(
                "Field `{}` in `{}` is missing `#[serde(default)]`. \
                 LangGraph states require this attribute on all fields to prevent \
                 state loss during resume operations. Please add `#[serde(default)]` \
                 to this field.",
                field_name, name
            );
            return syn::Error::new_spanned(field, error_msg).to_compile_error().into();
        }
    }

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
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let args = parse_macro_input!(attr as ToolMacroArgs);
    impl_tool_macro(&args.name, &args.description, &func)
}

struct ToolMacroArgs {
    name: Option<Lit>,
    description: Option<Lit>,
}

impl syn::parse::Parse for ToolMacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self { name: None, description: None });
        }
        let name: Lit = input.parse()?;
        let description = if input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
            Some(input.parse()?)
        } else {
            None
        };
        Ok(Self { name: Some(name), description })
    }
}

fn impl_tool_macro(name_lit: &Option<Lit>, desc_lit: &Option<Lit>, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();

    let tool_name = if let Some(Lit::Str(s)) = name_lit {
        s.value()
    } else {
        fn_name_str.clone()
    };

    // Extract parameter descriptions from @param lines in doc comments.
    let param_descs = extract_param_descs(func);

    let description = if let Some(desc) = desc_lit {
        match desc {
            Lit::Str(s) => s.value(),
            _ => panic!("description must be a string literal"),
        }
    } else {
        let mut extracted_desc = String::new();
        for attr in &func.attrs {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                            let doc_str = lit_str.value();
                            let trimmed = doc_str.trim();
                            // Skip @param lines — they are for schema, not description.
                            if trimmed.starts_with("@param ") {
                                continue;
                            }
                            if !extracted_desc.is_empty() {
                                extracted_desc.push_str(" ");
                            }
                            extracted_desc.push_str(trimmed);
                        }
                    }
                }
            }
        }
        extracted_desc
    };

    let struct_name_str = to_camel_case(&fn_name_str);
    let struct_name = syn::Ident::new(&struct_name_str, fn_name.span());

    let params: Vec<_> = func.sig.inputs.iter().filter_map(|arg| {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                return Some((pat_ident.ident.clone(), (*pat_type.ty).clone()));
            }
        }
        None
    }).collect();

    let properties: Vec<proc_macro2::TokenStream> = params.iter().map(|(name, ty)| {
        let name_str = name.to_string();
        let actual_ty = if is_option(ty) { extract_type_from_option(ty) } else { ty };
        let json_type = rust_type_to_json_type(actual_ty);
        if let Some(d) = param_descs.get(&name_str) {
            quote! {
                (#name_str, serde_json::json!({"type": #json_type, "description": #d}))
            }
        } else {
            quote! {
                (#name_str, serde_json::json!({"type": #json_type}))
            }
        }
    }).collect();

    let required: Vec<String> = params.iter()
        .filter(|(_, ty)| !is_option(ty))
        .map(|(name, _)| name.to_string())
        .collect();

    let extractions: Vec<proc_macro2::TokenStream> = params.iter().map(|(name, ty)| {
        let name_str = name.to_string();
        let err_invalid = format!("invalid parameter '{}': {{}}", name_str);
        
        if is_option(ty) {
            quote! {
                let #name: #ty = match args.get(#name_str) {
                    Some(v) => serde_json::from_value(v.clone())
                        .map_err(|e| langgraph_prebuilt::ToolError::InvalidArgs(format!(#err_invalid, e)))?,
                    None => None,
                };
            }
        } else {
            let err_missing = format!("missing required parameter '{}'", name_str);
            quote! {
                let #name: #ty = serde_json::from_value(
                    args.get(#name_str)
                        .cloned()
                        .ok_or_else(|| langgraph_prebuilt::ToolError::InvalidArgs(#err_missing.to_string()))?
                ).map_err(|e| langgraph_prebuilt::ToolError::InvalidArgs(
                    format!(#err_invalid, e)
                ))?;
            }
        }
    }).collect();

    let param_names: Vec<_> = params.iter().map(|(name, _)| name.clone()).collect();

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

    let is_async = func.sig.asyncness.is_some();

    let await_tokens = if is_async {
        quote! { .await }
    } else {
        quote! {}
    };

    let invoke_body = if is_result_return {
        quote! {
            #(#extractions)*
            let result = #fn_name(#(#param_names),*)#await_tokens;
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
            let result = #fn_name(#(#param_names),*)#await_tokens;
            serde_json::to_value(result).map_err(|e| langgraph_prebuilt::ToolError::Execution(
                format!("failed to serialize result: {}", e)
            ))
        }
    };

    let trait_methods = if is_async {
        quote! {
            fn invoke(
                &self,
                _args: &serde_json::Value,
                _config: &langgraph_checkpoint::config::RunnableConfig,
            ) -> Result<serde_json::Value, langgraph_prebuilt::ToolError> {
                Err(langgraph_prebuilt::ToolError::Execution(
                    "This tool is asynchronous and must be invoked with ainvoke".to_string()
                ))
            }

            async fn ainvoke(
                &self,
                args: &serde_json::Value,
                _config: &langgraph_checkpoint::config::RunnableConfig,
            ) -> Result<serde_json::Value, langgraph_prebuilt::ToolError> {
                #invoke_body
            }
        }
    } else {
        quote! {
            fn invoke(
                &self,
                args: &serde_json::Value,
                _config: &langgraph_checkpoint::config::RunnableConfig,
            ) -> Result<serde_json::Value, langgraph_prebuilt::ToolError> {
                #invoke_body
            }
        }
    };

    let expanded = quote! {
        #func
        pub struct #struct_name;
        impl #struct_name {
            pub fn new() -> Self { Self }
        }
        impl Default for #struct_name {
            fn default() -> Self { Self }
        }
        #[async_trait::async_trait]
        impl langgraph_prebuilt::BaseTool for #struct_name {
            fn name(&self) -> &str { #tool_name }
            fn description(&self) -> &str { #description }
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
            #trait_methods
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

// ============================================================================
// #[derive(Traceable)]
// ============================================================================
#[proc_macro_derive(Traceable)]
pub fn derive_traceable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_traceable(&input)
}

fn impl_traceable(input: &DeriveInput) -> TokenStream {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            pub fn tracing_context() -> langgraph_tracing::TracingContext {
                langgraph_tracing::TracingContext::new()
            }
        }
    };
    TokenStream::from(expanded)
}

fn is_option(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
}

fn extract_type_from_option(ty: &syn::Type) -> &syn::Type {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if segment.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                        return inner_ty;
                    }
                }
            }
        }
    }
    ty
}

fn extract_param_descs(func: &ItemFn) -> std::collections::HashMap<String, String> {
    let mut descs = std::collections::HashMap::new();
    for attr in &func.attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &attr.meta {
            if let syn::Expr::Lit(expr_lit) = &nv.value {
                if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                    let line = lit_str.value();
                    let trimmed = line.trim();
                    // Parse "@param name description"
                    if let Some(rest) = trimmed.strip_prefix("@param ") {
                        let rest = rest.trim_start();
                        if let Some(space_idx) = rest.find(char::is_whitespace) {
                            let name = rest[..space_idx].to_string();
                            let desc = rest[space_idx..].trim().to_string();
                            if !desc.is_empty() {
                                descs.insert(name, desc);
                            }
                        }
                    }
                }
            }
        }
    }
    descs
}
