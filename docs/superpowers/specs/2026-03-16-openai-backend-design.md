# OpenAI Backend Design

## Goal

Add a first-class `openai` worker backend so the project can call OpenAI's GPT-5.2 vision model directly without requiring a separate `/infer` proxy service.

## Scope

- Add a new backend type: `openai`
- Add OpenAI-specific configuration under `worker.openai`
- Implement a backend that submits multi-image requests to OpenAI Responses API
- Parse structured model output into the existing `vlm_json` shape
- Update docs and examples for configuration and installation

## Non-Goals

- Changing the server queue or segmentation algorithm
- Replacing the existing `remote_api` backend
- Supporting ChatGPT product UI integrations
- Supporting `gpt-5.2-pro`

## Architecture

The server and worker contract remains unchanged. The worker continues to fetch image windows, build the existing switch-detection prompt, and submit a `vlm_json` payload back to the server. The only change in the runtime path is that the worker can now instantiate an OpenAI-backed implementation of the `VLMBackend` interface.

The OpenAI backend will encode each frame as JPEG, submit a single Responses API request containing one text prompt plus multiple image inputs, and request a strict JSON schema with `thought`, `transitions`, and `instructions`. If the request fails or the response cannot be parsed into the expected shape, the backend returns `{}` so the existing retry behavior continues to work unchanged.

## Configuration

Add `worker.openai` with:

- `api_key`
- `model` defaulting to `gpt-5.2`
- `base_url` defaulting to `https://api.openai.com/v1`
- `timeout_sec`
- `organization`
- `project`
- `reasoning_effort`
- `max_output_tokens`
- `jpeg_quality`

`api_key` should fall back to `OPENAI_API_KEY` when omitted in YAML.

## Error Handling

- Non-200 API responses log status details and return `{}`
- Invalid JSON or schema mismatches return `{}`
- Missing API key raises a clear configuration error during backend creation
- Existing worker retries remain the recovery path

## Testing

Add tests for:

- Config validation accepting `backend: openai`
- Backend request construction and schema parsing
- Environment fallback for `OPENAI_API_KEY`
- Response normalization from structured JSON into the current backend contract

## Documentation

Update the example config and README files to explain:

- how to select `worker.backend: openai`
- which config fields are required
- how to provide an API key
- that GPT-5.2 is the intended target model
