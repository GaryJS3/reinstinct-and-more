//! OpenAI Chat Completions tool-call protocol types and validation.
//!
//! This module deliberately owns only the wire/domain layer.  Model-specific
//! rendering and parsing live in `chat.rs`; the HTTP server should not need to
//! know whether Qwen emits XML, JSON, or another native tool envelope.

use super::json::Json;

const MAX_TOOLS: usize = 128;
const MAX_TOOL_NAME: usize = 64;
const MAX_TOOL_DESCRIPTION: usize = 16 * 1024;
const MAX_TOOL_SCHEMA_BYTES: usize = 256 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Json,
    pub strict: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ToolCall {
    pub id: String,
    pub name: String,
    /// OpenAI carries arguments as a JSON string, including in the
    /// non-streaming response.  Keeping that representation here prevents
    /// accidental lossy re-serialization between turns.
    pub arguments: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ToolChoice {
    None,
    Auto,
    Required,
    Function(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChatToolOptions {
    pub tools: Vec<FunctionTool>,
    pub choice: ToolChoice,
    pub parallel: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParsedAssistant {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl Default for ChatToolOptions {
    fn default() -> Self {
        Self {
            tools: Vec::new(),
            choice: ToolChoice::Auto,
            parallel: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ToolChatMessage {
    System(String),
    User(String),
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

pub(crate) fn parse_options(j: &Json) -> Result<ChatToolOptions, String> {
    for key in ["functions", "function_call"] {
        if j.get(key).is_some() {
            return Err(format!(
                "field '{key}' is not supported yet; use modern 'tools'"
            ));
        }
    }
    let mut options = ChatToolOptions::default();
    if let Some(value) = j.get("tools") {
        let values = match value {
            Json::Arr(values) => values,
            _ => return Err("field 'tools' must be an array".into()),
        };
        if values.len() > MAX_TOOLS {
            return Err(format!(
                "field 'tools' supports at most {MAX_TOOLS} functions"
            ));
        }
        for (i, value) in values.iter().enumerate() {
            let kind = value
                .get("type")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("tools[{i}]: missing string 'type'"))?;
            if kind != "function" {
                return Err(format!(
                    "tools[{i}]: unsupported type '{kind}' (want function)"
                ));
            }
            let function = value
                .get("function")
                .ok_or_else(|| format!("tools[{i}]: missing object 'function'"))?;
            let name = function
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("tools[{i}].function: missing string 'name'"))?;
            validate_name(name).map_err(|e| format!("tools[{i}].function.name: {e}"))?;
            if options.tools.iter().any(|tool| tool.name == name) {
                return Err(format!(
                    "tools[{i}].function.name: duplicate function '{name}'"
                ));
            }
            let description = match function.get("description") {
                None => None,
                Some(Json::Str(value)) if value.len() <= MAX_TOOL_DESCRIPTION => {
                    Some(value.clone())
                }
                Some(Json::Str(_)) => {
                    return Err(format!(
                        "tools[{i}].function.description exceeds {MAX_TOOL_DESCRIPTION} bytes"
                    ))
                }
                Some(_) => return Err(format!("tools[{i}].function.description must be a string")),
            };
            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| Json::Obj(vec![("type".into(), Json::Str("object".into()))]));
            if !matches!(parameters, Json::Obj(_)) {
                return Err(format!("tools[{i}].function.parameters must be an object"));
            }
            if parameters.to_string().len() > MAX_TOOL_SCHEMA_BYTES {
                return Err(format!(
                    "tools[{i}].function.parameters exceeds {MAX_TOOL_SCHEMA_BYTES} bytes"
                ));
            }
            let strict = match function.get("strict") {
                None => None,
                Some(Json::Bool(value)) => Some(*value),
                Some(_) => return Err(format!("tools[{i}].function.strict must be a boolean")),
            };
            options.tools.push(FunctionTool {
                name: name.to_owned(),
                description,
                parameters,
                strict,
            });
        }
    }

    options.choice = match j.get("tool_choice") {
        None => {
            if options.tools.is_empty() {
                ToolChoice::None
            } else {
                ToolChoice::Auto
            }
        }
        Some(Json::Str(value)) => match value.as_str() {
            "none" => ToolChoice::None,
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Required,
            _ => return Err("field 'tool_choice' must be none, auto, or required".into()),
        },
        Some(Json::Obj(_)) => {
            let kind = j
                .get("tool_choice")
                .and_then(|v| v.get("type"))
                .and_then(Json::as_str)
                .ok_or("field 'tool_choice.type' must be 'function'")?;
            if kind != "function" {
                return Err("field 'tool_choice.type' must be 'function'".into());
            }
            let name = j
                .get("tool_choice")
                .and_then(|v| v.get("function"))
                .and_then(|v| v.get("name"))
                .and_then(Json::as_str)
                .ok_or("field 'tool_choice.function.name' must be a string")?;
            validate_name(name).map_err(|e| format!("field 'tool_choice.function.name': {e}"))?;
            ToolChoice::Function(name.to_owned())
        }
        Some(_) => return Err("field 'tool_choice' must be a string or function object".into()),
    };

    if let ToolChoice::Function(name) = &options.choice {
        if !options.tools.iter().any(|tool| &tool.name == name) {
            return Err(format!("tool_choice references unknown function '{name}'"));
        }
    }
    if matches!(
        options.choice,
        ToolChoice::Required | ToolChoice::Function(_)
    ) && options.tools.is_empty()
    {
        return Err("tool_choice requires at least one function in tools".into());
    }
    options.parallel = match j.get("parallel_tool_calls") {
        None => true,
        Some(Json::Bool(value)) => *value,
        Some(_) => return Err("field 'parallel_tool_calls' must be a boolean".into()),
    };
    Ok(options)
}

pub(crate) fn parse_assistant_tool_calls(
    message: &Json,
    index: usize,
) -> Result<Vec<ToolCall>, String> {
    let values = message
        .get("tool_calls")
        .ok_or_else(|| format!("messages[{index}]: missing 'tool_calls'"))?;
    let values = match values {
        Json::Arr(values) => values,
        _ => return Err(format!("messages[{index}].tool_calls must be an array")),
    };
    let mut calls = Vec::with_capacity(values.len());
    for (call_index, value) in values.iter().enumerate() {
        let id = value.get("id").and_then(Json::as_str).ok_or_else(|| {
            format!("messages[{index}].tool_calls[{call_index}]: missing string 'id'")
        })?;
        if id.is_empty() || id.len() > 256 {
            return Err(format!(
                "messages[{index}].tool_calls[{call_index}].id has invalid length"
            ));
        }
        let kind = value.get("type").and_then(Json::as_str).ok_or_else(|| {
            format!("messages[{index}].tool_calls[{call_index}]: missing string 'type'")
        })?;
        if kind != "function" {
            return Err(format!(
                "messages[{index}].tool_calls[{call_index}]: unsupported type '{kind}'"
            ));
        }
        let function = value.get("function").ok_or_else(|| {
            format!("messages[{index}].tool_calls[{call_index}]: missing object 'function'")
        })?;
        let name = function.get("name").and_then(Json::as_str).ok_or_else(|| {
            format!("messages[{index}].tool_calls[{call_index}].function: missing string 'name'")
        })?;
        validate_name(name).map_err(|e| {
            format!("messages[{index}].tool_calls[{call_index}].function.name: {e}")
        })?;
        let arguments = function
            .get("arguments")
            .and_then(Json::as_str)
            .ok_or_else(|| {
                format!(
            "messages[{index}].tool_calls[{call_index}].function: missing string 'arguments'")
            })?;
        Json::parse(arguments).map_err(|e| {
            format!(
            "messages[{index}].tool_calls[{call_index}].function.arguments is not valid JSON: {e}")
        })?;
        calls.push(ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        });
    }
    if calls.is_empty() {
        return Err(format!("messages[{index}].tool_calls must not be empty"));
    }
    Ok(calls)
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > MAX_TOOL_NAME {
        return Err(format!("must be 1-{MAX_TOOL_NAME} bytes"));
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err("may contain only letters, digits, underscores, and hyphens".into());
    }
    Ok(())
}

/// Parse the XML envelope emitted by the Qwen 3.5/3.6 tool template.
///
/// The model emits parameter values as readable XML bodies, while OpenAI
/// requires the response arguments to be one JSON string.  The function
/// schema is used to retain scalar types (for example, `3` as a number rather
/// than the string `"3"`); object and array bodies are parsed as JSON.
pub(crate) fn parse_qwen_output(
    text: &str,
    options: &ChatToolOptions,
    request_id: u64,
) -> Result<ParsedAssistant, String> {
    let opener = "<tool_call>";
    let closer = "</tool_call>";
    let mut cursor = 0;
    let mut content = String::new();
    let mut calls = Vec::new();
    while let Some(relative) = text[cursor..].find(opener) {
        let start = cursor + relative;
        content.push_str(&text[cursor..start]);
        let body_start = start + opener.len();
        let relative_end = text[body_start..]
            .find(closer)
            .ok_or("qwen tool call is missing </tool_call>")?;
        let end = body_start + relative_end;
        let call = parse_qwen_call_body(&text[body_start..end], options, request_id, calls.len())?;
        calls.push(call);
        cursor = end + closer.len();
    }
    content.push_str(&text[cursor..]);
    if calls.is_empty() {
        if matches!(
            options.choice,
            ToolChoice::Required | ToolChoice::Function(_)
        ) {
            return Err("model did not emit the required tool call".into());
        }
        return Ok(ParsedAssistant {
            content,
            tool_calls: calls,
        });
    }
    if !options.parallel && calls.len() > 1 {
        return Err("model emitted multiple tool calls while parallel_tool_calls=false".into());
    }
    if matches!(options.choice, ToolChoice::None) {
        return Err("model emitted a tool call while tool_choice=none".into());
    }
    if let ToolChoice::Function(expected) = &options.choice {
        if calls.iter().any(|call| &call.name != expected) {
            return Err(format!(
                "model emitted a tool other than required function '{expected}'"
            ));
        }
    }
    Ok(ParsedAssistant {
        content: content.trim().to_owned(),
        tool_calls: calls,
    })
}

fn parse_qwen_call_body(
    body: &str,
    options: &ChatToolOptions,
    request_id: u64,
    call_index: usize,
) -> Result<ToolCall, String> {
    let function_open = "<function=";
    let function_start = body
        .find(function_open)
        .ok_or("qwen tool call is missing <function=...>")?
        + function_open.len();
    let name_end = body[function_start..]
        .find('>')
        .ok_or("qwen tool call has an unterminated function name")?
        + function_start;
    let name = body[function_start..name_end].trim();
    validate_name(name).map_err(|e| format!("qwen tool function '{name}': {e}"))?;
    let function_body_start = name_end + 1;
    let function_body_end = body[function_body_start..]
        .find("</function>")
        .ok_or("qwen tool call is missing </function>")?
        + function_body_start;
    if !options.tools.iter().any(|tool| tool.name == name) {
        return Err(format!("model emitted unknown function '{name}'"));
    }
    let function_body = &body[function_body_start..function_body_end];
    let mut fields = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = function_body[cursor..].find("<parameter=") {
        let start = cursor + relative;
        let name_start = start + "<parameter=".len();
        let name_end = function_body[name_start..]
            .find('>')
            .ok_or("qwen tool parameter has an unterminated name")?
            + name_start;
        let parameter_name = function_body[name_start..name_end].trim();
        if !parameter_name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(format!(
                "qwen tool parameter '{parameter_name}' has an invalid name"
            ));
        }
        let value_start = name_end + 1;
        let value_end = function_body[value_start..]
            .find("</parameter>")
            .ok_or(format!(
                "qwen tool parameter '{parameter_name}' is missing </parameter>"
            ))?
            + value_start;
        let raw = function_body[value_start..value_end]
            .trim_matches(|c: char| c == '\n' || c == '\r' || c == ' ' || c == '\t');
        let schema = options
            .tools
            .iter()
            .find(|tool| tool.name == name)
            .and_then(|tool| tool.parameters.get("properties"))
            .and_then(|properties| properties.get(parameter_name));
        let value = parse_parameter_value(raw, schema)
            .map_err(|e| format!("qwen tool parameter '{parameter_name}': {e}"))?;
        fields.push((parameter_name.to_owned(), value));
        cursor = value_end + "</parameter>".len();
    }
    let arguments = Json::Obj(fields).to_string();
    Json::parse(&arguments).map_err(|e| format!("qwen tool arguments are not valid JSON: {e}"))?;
    Ok(ToolCall {
        id: format!("call-{request_id}-{call_index}"),
        name: name.to_owned(),
        arguments,
    })
}

fn parse_parameter_value(raw: &str, schema: Option<&Json>) -> Result<Json, String> {
    let declared_type = schema
        .and_then(|value| value.get("type"))
        .and_then(Json::as_str);
    if declared_type == Some("string") || declared_type.is_none() {
        if raw.starts_with('{') || raw.starts_with('[') {
            return Json::parse(raw);
        }
        return Ok(Json::Str(raw.to_owned()));
    }
    Json::parse(raw).map_err(|e| format!("expected JSON value for type '{declared_type:?}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(extra: &str) -> Json {
        Json::parse(&format!(r#"{{"tools":[{{"type":"function","function":{{"name":"read","description":"Read a file","parameters":{{"type":"object","properties":{{"path":{{"type":"string"}}}},"required":["path"]}}}}}}],{extra}}}"#)).unwrap()
    }

    #[test]
    fn parses_function_tools_and_defaults_to_auto() {
        let options = parse_options(&request("\"tool_choice\":\"auto\"")).unwrap();
        assert_eq!(options.tools.len(), 1);
        assert_eq!(options.choice, ToolChoice::Auto);
        assert!(options.parallel);
    }

    #[test]
    fn parses_named_choice_and_rejects_unknown_name() {
        let options = parse_options(&request(
            "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"read\"}}",
        ))
        .unwrap();
        assert_eq!(options.choice, ToolChoice::Function("read".into()));
        assert!(parse_options(&request(
            "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"write\"}}"
        ))
        .is_err());
    }

    #[test]
    fn empty_tools_is_supported_and_means_no_tools() {
        let j = Json::parse(r#"{"tools":[]}"#).unwrap();
        let options = parse_options(&j).unwrap();
        assert!(options.tools.is_empty());
        assert_eq!(options.choice, ToolChoice::None);
    }

    #[test]
    fn parses_qwen_xml_call_into_openai_arguments() {
        let options = parse_options(&request("\"tool_choice\":\"required\"")).unwrap();
        let parsed = parse_qwen_output(
            "before <tool_call>\n<function=read>\n<parameter=path>\nhello.txt\n</parameter>\n</function>\n</tool_call>",
            &options, 7).unwrap();
        assert_eq!(parsed.content, "before");
        assert_eq!(parsed.tool_calls[0].id, "call-7-0");
        assert_eq!(parsed.tool_calls[0].name, "read");
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"path":"hello.txt"}"#);
    }

    #[test]
    fn rejects_model_tool_call_when_choice_is_none() {
        let mut options = parse_options(&request("\"tool_choice\":\"none\"" )).unwrap();
        let error = parse_qwen_output(
            "<tool_call>\n<function=read>\n</function>\n</tool_call>",
            &options, 8).unwrap_err();
        assert!(error.contains("tool_choice=none"));
        options.choice = ToolChoice::Auto;
    }
}
