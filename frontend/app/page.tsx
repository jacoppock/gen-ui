"use client";
import Chat from "@/components/prebuilt/chat";
import { useState } from "react"; // Add this import

const AgentResponsePanel = ({ response }: { response: any }) => {
  // Check if response is an object and has a 'ui' property
  const content = response && typeof response === 'object' && 'ui' in response ? response.ui : response;

  return (
    <div className="agent-response-panel">
      {content ? (
        // Render the content if it's valid
        <div>{content}</div>
      ) : (
        <div>No response yet</div>
      )}
    </div>
  );
};

export default function Home() {
  const [response, setResponse] = useState<string | null>(null); // Add state for response

  return (
    <main className="flex h-screen">
      <div className="w-1/2 flex flex-col items-center justify-between px-4">
        <div className="w-full min-w-[600px] flex flex-col gap-4">
          <p className="text-[28px] text-center font-medium">
            Generative UI with{" "}
            <a
              href="https://github.com/langchain-ai/langchainjs"
              target="_blank"
              className="text-blue-600 hover:underline hover:underline-offset-2"
            >
              Automated Health 🩻
            </a>
          </p>
          <Chat response={response} setResponse={setResponse} /> {/* Pass response and setter */}
        </div>
      </div>
      <div className="w-1/2 p-4">
        <AgentResponsePanel response={response} /> {/* Render the response panel */}
      </div>
    </main>
  );
}
