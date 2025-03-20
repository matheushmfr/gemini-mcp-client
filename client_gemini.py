import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

# from anthropic import Anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        """Initialize the Vertex AI client for Google Gemini."""
        self.genai_client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "data-science-hml"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

        self.model = "gemini-2.0-flash-001"

        self.generate_config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""

        # Keep track of the conversation
        self.conversation_history = []

        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Get available tools
        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        # Format tools for Gemini

        tools_description = "Available tools:\n"
        for tool in available_tools:
            tools_description += f"- {tool['name']}: {tool['description']}\n"
            tools_description += (
                f"  Input schema: {json.dumps(tool['input_schema'])}\n\n"
            )

        # Initial Gemini API call
        gemini_prompt = f"""
        You are a helpful assistant that can use tools to answer questions.
        When you need to use a tool, format your response as follows:
        
        <tool_call>
        {{
            "name": "tool_name",
            "input": {{
                "parameter1": "value1",
                "parameter2": "value2"
            }}
        }}
        </tool_call>
        
        After using a tool, provide a natural language response based on the tool results.
        
        {tools_description}

        - If the user query does not indicate the need to use a tool, provide a natural language response.
        - Remember to wrap all tool calls with "<tool_call>" {{JSON representing the tool call}} "</tool_call>" tags.
        - It's important to use the XML-like syntax for the tags, with "<TAG_NAME>" and "</TAG_NAME>" tags.
        
        
        User query: {query}
        """

        # Create Gemini content
        contents = [types.Content(role="user", parts=[types.Part(text=gemini_prompt)])]

        # Get response from Gemini
        response = await self._generate_content(contents)

        # Parse response for tool calls
        final_text = []
        tool_calls = self._extract_tool_calls(response)
        print(tool_calls)

        if tool_calls:
            for tool_call in tool_calls:
                try:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["input"]

                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(
                        f"[Calling tool {tool_name} with args {tool_args}]"
                    )

                    # Use the tool result to get a final response
                    tool_result_prompt = f"""
                    You previously used the tool {tool_name} to answer the query: {query}
                    
                    The tool returned this result: {result.content}
                    
                    Please provide a helpful response based on this information.
                    """

                    # Create Gemini content for tool result
                    tool_contents = [
                        types.Content(
                            role="user", parts=[types.Part(text=tool_result_prompt)]
                        )
                    ]

                    # Get final response from Gemini
                    final_response = await self._generate_content(tool_contents)
                    final_text.append(final_response)
                except Exception as e:
                    final_text.append(f"Error executing tool {tool_name}: {str(e)}")
        else:
            # If no tool calls, just use the response directly
            print("\n\nNo tool calls were detected.\n\n")
            final_text.append(response)

        return "\n".join(final_text)

    async def _generate_content(self, contents: List[types.Content]) -> str:
        """Generate content using Gemini"""
        response = self.genai_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self.generate_config,
        )
        return response.text

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from Gemini response"""
        tool_calls = []

        # Look for tool call format with the format
        # <tool_call>
        # {
        #   "name": "tool_name",
        #   "input": {
        #     "parameter1": "value1",
        #     "parameter2": "value2"
        #   }
        # }
        # </tool_call>

        pattern = r"<?tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                print(f"Failed to parse tool call: {match}")

        return tool_calls

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client with Gemini Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
