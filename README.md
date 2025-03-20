# Gemini MCP Client

A Model Context Protocol (MCP) client implementation that connects to MCP servers and uses Google's Gemini 2.0 Flash model for intelligent function calling.

## Overview

This project provides an MCP client that can integrate with any MCP-compatible server. It uses Google's Gemini 2.0 Flash model to interpret user queries and intelligently call server-defined tools through the MCP protocol.

### Features

- Connect to any MCP server (Python or Node.js)
- Automatic conversion of MCP tools to Gemini function declarations
- Native function calling using Gemini's built-in capabilities
- Smart context handling using Gemini's chat interface
- Interactive command-line interface
- Helpful system instructions for balanced tool usage

## Requirements

- Google Cloud project with Gemini API enabled

## Setting Up Your Environment

Clone the repository and set up the environment using uv:

```bash
# Clone the repository
git clone https://github.com/yourusername/gemini-mcp-client.git
cd gemini-mcp-client

# Create and activate virtual environment with uv
uv venv
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# Install dependencies using uv sync
uv sync
```

> **About `uv sync`**: The `uv sync` command installs all dependencies specified in your project's pyproject.toml file. Unlike traditional pip install methods, `uv sync` is significantly faster as it resolves dependencies in parallel and caches them efficiently. It ensures that all your dependencies are installed with the exact versions specified in your project, maintaining consistency across different environments.


## Setting Up Authentication

### Google Cloud Authentication

You'll need to set up Google Cloud authentication for your project:

```bash
# Create .env file
touch .env
```

Add your Google Cloud project details to the .env file:

```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

Add .env to your .gitignore:

```bash
echo ".env" >> .gitignore
```

Make sure you authenticate with Google Cloud:

```bash
gcloud auth application-default login
```

## Running the Client

To run your client with any MCP server:

```bash
# For Python servers
python client.py path/to/server.py

# For Node.js servers
python client.py path/to/build/index.js
```

The client will:
1. Connect to the specified server
2. List available tools
3. Start an interactive chat session where you can:
   - Enter queries
   - See tool executions
   - Get responses from Gemini

## Quickstart: Setting Up a Test Server

To test this client, you'll need an MCP server to connect to. Follow these steps:

1. Set up a test server by following the official MCP server quickstart guide: [MCP Server Quickstart](https://modelcontextprotocol.io/quickstart/server)
2. This will walk you through creating a simple weather server that provides forecast tools
3. Once you've completed the server setup, you can connect the Gemini client to it:

```bash
# Make sure your virtual environment is activated
uv run client_gemini_function_call.py path/to/your/server/weather.py
```

## Client Implementation Variants

This repository provides two different approaches to implementing an MCP client with Gemini:

### 1. Prompt-Based Tool Parsing (`client_gemini.py`)

The first approach uses a simple prompting strategy to extract tool calls:

```bash
uv run client_gemini.py path/to/your/server.py
```

**Key features:**
- Formats tool descriptions directly in the prompt
- Uses custom regex patterns to extract tool calls from Gemini's response
- Manually handles the conversation context
- Uses "generate_content" implementation in google's genai SDK.


### 2. Native Function Calling (`client_gemini_function_call.py`)

The second approach leverages Gemini's built-in function calling capabilities:

```bash
uv run client_gemini_function_call.py path/to/your/server.py
```

**Key features:**
- Converts MCP tools to Gemini's function declarations format
- Uses Gemini's native function calling API
- Automatically handles tool execution and context management
- Provides more reliable tool integration
- Uses the "chat" implementation in google's genai SDK.


## How It Works

When you submit a query:

1. The client gets the list of available tools from the server
2. Your query is sent to Gemini along with tool descriptions
3. Gemini decides which tools (if any) to use based on your query
4. The client executes any requested tool calls through the server
5. Results are sent back to Gemini
6. Gemini provides a natural language response
7. The response is displayed to you

## Key Components

The client consists of several key components:

1. **Server Connection**: Handles connecting to MCP servers and listing available tools
2. **Tool Conversion**: Automatically converts MCP tool schemas to Gemini function declarations
3. **Query Processing**: Manages the flow of requests and responses between you, Gemini, and tools
4. **Interactive Interface**: Provides a simple command-line interface for interacting with the system


## Troubleshooting

### Authentication Issues
- Ensure you've run `gcloud auth application-default login`
- Verify your project ID and location are correct
- Check that your Google Cloud project has the necessary API enabled

### Server Path Issues
- Double-check the path to your server script is correct
- Use the absolute path if the relative path isn't working
- For Windows users, make sure to use forward slashes (/) or escaped backslashes (\\) in the path

### Common Error Messages
- `FileNotFoundError`: Check your server path
- `Connection refused`: Ensure the server is running and the path is correct
- `Tool execution failed`: Verify any required environment variables for tools are set