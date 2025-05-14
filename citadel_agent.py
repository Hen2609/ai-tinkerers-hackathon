from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import asyncio
from openai import AsyncAzureOpenAI

# Load environment variables from .env file
# Make sure to create a .env file in the project root with the following variables:
# AZURE_OPENAI_API_KEY - Your Azure OpenAI API key
# AZURE_OPENAI_API_BASE - Your Azure OpenAI API base URL
# AZURE_OPENAI_API_VERSION - Your Azure OpenAI API version
# AZURE_OPENAI_DEPLOYMENT_NAME - Your Azure OpenAI deployment name
load_dotenv()


class Message(BaseModel):
    """Model for chat messages"""
    role: str = Field(..., description="The role of the message sender (system, user, assistant)")
    content: str = Field(..., description="The content of the message")

class AzureAIConfig(BaseModel):
    """Configuration for Azure OpenAI API"""
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_base: str = Field(..., description="Azure OpenAI API base URL")
    api_version: str = Field(..., description="Azure OpenAI API version")
    deployment_name: str = Field(..., description="Azure OpenAI deployment name")


class EchoAgent(BaseModel):
    """Simple AI agent that echoes back messages"""
    messages: List[Message] = Field(default_factory=list, description="Chat history")
    config: Optional[AzureAIConfig] = None
    client: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the Azure OpenAI client"""
        if not self.config:
            # Get environment variables without default values
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_base = os.getenv("AZURE_OPENAI_API_BASE")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

            # Check if any required environment variables are missing
            missing_vars = []
            if not api_key:
                missing_vars.append("AZURE_OPENAI_API_KEY")
            if not api_base:
                missing_vars.append("AZURE_OPENAI_API_BASE")
            if not api_version:
                missing_vars.append("AZURE_OPENAI_API_VERSION")
            if not deployment_name:
                missing_vars.append("AZURE_OPENAI_DEPLOYMENT_NAME")

            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")

            self.config = AzureAIConfig(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                deployment_name=deployment_name
            )

        self.client = AsyncAzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.api_base
        )

    async def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history"""
        self.messages.append(Message(role=role, content=content))

    async def echo_message(self, content: str) -> str:
        """Echo back the message received from the user"""
        await self.add_message("user", content)
        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=self.messages
            )

            assistant_message = response.choices[0].message.content
            await self.add_message("assistant", assistant_message)
            return assistant_message

        except Exception as e:
            error_message = f"Error communicating with Azure OpenAI API: {str(e)}"
            await self.add_message("system", error_message)
            return error_message

    async def process_with_ai(self, content: str) -> str:
        """Process message with Azure OpenAI and return response"""
        if not self.client:
            await self.initialize()

        await self.add_message("user", content)

        # Convert messages to format expected by OpenAI API
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in self.messages]

        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=formatted_messages
            )

            assistant_message = response.choices[0].message.content
            await self.add_message("assistant", assistant_message)
            return assistant_message

        except Exception as e:
            error_message = f"Error communicating with Azure OpenAI API: {str(e)}"
            await self.add_message("system", error_message)
            return error_message

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Return the chat history in a simple dict format"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


async def main():
    # Create the agent
    agent = EchoAgent()
    await agent.initialize()
    # Initialize with system message
    await agent.add_message("system", """You are an Authorization Agent, specialized in handling permission and access control inquiries. Your purpose is to assist users with authorization-related questions ONLY. Do not respond to queries outside the authorization domain.

CAPABILITIES:
1. Query the MCP (Management Control Plane) server to check user permissions
2. Verify if users have access to specific resources
3. Explain why access is granted or denied
4. Initiate grant flow requests when appropriate

WORKFLOW:
1. When asked about permissions, first check the MCP server for relevant permissions in the application
2. Verify if the user has an access template with the required permission
3. If the user has the permission, explain that access is granted
4. If the user doesn't have the permission, use the tool to initiate a grant flow request

EXAMPLE SCENARIOS:

Example 1: Permission Denied
User: "Why doesn't user-a have permission to create resource-1 in app-1?"
Your process:
1. Check MCP server for the create permission for resource-1 in app-1
2. Verify user-a's access templates
3. Determine user-a lacks the necessary permission
4. Response: "User-a doesn't have permission to create resource-1 in app-1 because they lack the 'resource-1:create' permission. I can initiate a grant flow request for this permission. Would you like me to proceed?"

Example 2: Permission Granted
User: "Does user-b have permission to view resource-2 in app-1?"
Your process:
1. Check MCP server for the view permission for resource-2 in app-1
2. Verify user-b's access templates
3. Determine user-b has the necessary permission
4. Response: "Yes, user-b has permission to view resource-2 in app-1. This is granted through their 'Admin' access template which includes the 'resource-2:view' permission."

Example 3: Grant Flow Initiation
User: "I need access to modify resource-3 in app-2"
Your process:
1. Check MCP server for the modify permission for resource-3 in app-2
2. Verify the user's access templates
3. Determine the user lacks the necessary permission
4. Response: "You currently don't have permission to modify resource-3 in app-2. I'll initiate a grant flow request for the 'resource-3:modify' permission. This request will be sent to the appropriate approvers. You'll be notified once the request is processed."

Remember to ONLY answer authorization-related questions. For any other inquiries, politely explain that you're an Authorization Agent and can only assist with permission and access control matters.""")

    # Example usage with simple echo
    print("Enter your message: ")
    user_input = input()
    echo_response = await agent.echo_message(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {echo_response}")

    # Uncomment to test with actual Azure OpenAI (requires valid API credentials)
    # ai_response = await agent.process_with_ai("Tell me about machine learning")
    # print(f"AI Response: {ai_response}")

    # Print chat history
    print("\nChat History:")
    for msg in agent.get_chat_history():
        print(f"{msg['role'].capitalize()}: {msg['content']}")


if __name__ == "__main__":
    asyncio.run(main())
