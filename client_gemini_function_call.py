import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Dict, Optional

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

        # Initialize Gemini client
        self.genai_client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        )
        self.model = "gemini-2.0-flash-001"

        # Configuration for Gemini (will be updated with tools)
        self.generate_config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
        )

        # Conversation history for chat
        self.chat = None

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

        # Set up Gemini tools based on MCP tools
        await self.connect_tools()

    async def connect_tools(self):
        """Convert MCP tools to Gemini function declarations"""
        # Get available tools from the MCP server
        response = await self.session.list_tools()
        mcp_tools = response.tools

        # Create function declarations for each MCP tool
        function_declarations = []
        for tool in mcp_tools:
            # Convert the MCP tool schema to a Gemini function declaration
            function_declaration = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=types.Schema(
                    type="OBJECT",
                    properties=self._convert_schema_properties(
                        tool.inputSchema.get("properties", {})
                    ),
                ),
            )
            function_declarations.append(function_declaration)

        # Create a Gemini Tool with all function declarations
        gemini_tool = types.Tool(function_declarations=function_declarations)

        # Define system instructions for the assistant
        system_instructions = """
        You are a helpful assistant with access to various tools. When a user's query requires the use of a tool,
        use the appropriate tool to address their needs. Do not suggest using a tool when it's not necessary and 
        answer queries not related to tools naturally.
        """

        self.generate_config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            tools=[gemini_tool],
            system_instruction=system_instructions,
        )

        # Initialize a chat session with the tools
        self.chat = self.genai_client.chats.create(
            model=self.model, config=self.generate_config
        )

        # Print available tools
        print("\nConnected to server with tools:", [tool.name for tool in mcp_tools])

    def _convert_schema_properties(self, properties: Dict) -> Dict[str, types.Schema]:
        """Convert JSON Schema properties to Gemini Schema properties"""
        converted_properties = {}

        for prop_name, prop_schema in properties.items():
            schema_type = self._map_json_schema_type(prop_schema.get("type", "string"))
            schema_description = prop_schema.get("description", "")

            # Handle nested objects
            if schema_type == "OBJECT" and "properties" in prop_schema:
                nested_properties = self._convert_schema_properties(
                    prop_schema["properties"]
                )
                converted_properties[prop_name] = types.Schema(
                    type=schema_type,
                    description=schema_description,
                    properties=nested_properties,
                )
            # Handle arrays
            elif schema_type == "ARRAY" and "items" in prop_schema:
                items_type = self._map_json_schema_type(
                    prop_schema["items"].get("type", "string")
                )
                converted_properties[prop_name] = types.Schema(
                    type=schema_type,
                    description=schema_description,
                    items=types.Schema(type=items_type),
                )
            # Handle simple types
            else:
                converted_properties[prop_name] = types.Schema(
                    type=schema_type, description=schema_description
                )

        return converted_properties

    def _map_json_schema_type(self, json_type: str) -> str:
        """Map JSON Schema types to Gemini Schema types"""
        type_mapping = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "object": "OBJECT",
            "array": "ARRAY",
        }
        return type_mapping.get(json_type, "STRING")

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available tools with native function calling"""
        try:
            # Send the query to Gemini
            response = self.chat.send_message(query)

            # Check if there are function calls
            final_text = []

            # If there are function calls, execute them
            if hasattr(response, "function_calls") and response.function_calls:
                for function_call in response.function_calls:
                    try:
                        tool_name = function_call.name
                        tool_args = function_call.args

                        # Log the function call
                        args_str = json.dumps(tool_args)
                        final_text.append(
                            f"[Calling tool {tool_name} with args {args_str}]"
                        )

                        # Execute the tool call via MCP
                        result = await self.session.call_tool(tool_name, tool_args)

                        # Send the function result back to Gemini
                        function_response = {
                            "name": tool_name,
                            "response": {"content": result.content},
                        }

                        message = f"""
                            You used the tool {tool_name} to answer the query: {query}
                    
                            The tool returned this result: {result.content}
                    
                            Please provide a helpful response based on this information.
                        """

                        # Get Gemini's response to the function results
                        response = self.chat.send_message(message)
                    except Exception as e:
                        error_message = f"Error executing tool {tool_name}: {str(e)}"
                        final_text.append(error_message)
                        print(error_message)

                        # Let Gemini know about the error
                        function_response = {
                            "name": tool_name,
                            "response": {"error": str(e)},
                        }
                        response = self.chat.send_message(
                            "", function_responses=[function_response]
                        )

            # Add the final response text
            final_text.append(response.text)
            return "\n".join(final_text)

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(error_message)
            return error_message

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client with Gemini Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break

                print("\nProcessing query...")
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
