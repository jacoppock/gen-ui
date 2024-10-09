"use client";
import Chat from "@/components/prebuilt/chat";
import { ResponsePanel } from "@/components/prebuilt/response-panel";
import { useState } from "react";

export default function Home() {
  const [response, setResponse] = useState<any>(null);

  return (
    <main className="flex h-screen bg-gray-100">
      <div className="w-1/2 p-6">
        <h1 className="text-3xl font-bold mb-6 text-gray-800">
          AI-assisted Patient Onboarding
        </h1>
        <Chat response={response} setResponse={setResponse} />
      </div>
      <div className="w-1/2 p-6 bg-white shadow-lg">
        <ResponsePanel dashboard={response?.ui} />
      </div>
    </main>
  );
}
