"use client";

import { EndpointsContext } from "@/app/agent";
import { useActions } from "@/utils/client";
import { Paperclip, Send } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { HumanMessageText } from "./message";

export interface ChatProps {
  response: any;
  setResponse: (response: any) => void;
}

function convertFileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64String = reader.result as string;
      resolve(base64String.split(",")[1]);
    };
    reader.onerror = (error) => {
      reject(error);
    };
    reader.readAsDataURL(file);
  });
}

function FileUploadMessage({ file }: { file: File }) {
  return (
    <div className="flex w-full max-w-fit ml-auto">
      <p className="text-sm text-gray-500">File uploaded: {file.name}</p>
    </div>
  );
}

const Chat: React.FC<ChatProps> = ({ response, setResponse }) => {
  const actions = useActions<typeof EndpointsContext>();
  const [elements, setElements] = useState<JSX.Element[]>([]);
  const [history, setHistory] = useState<[role: string, content: string][]>([]);
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File>();
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const handleAgentResponse = (newResponse: any) => {
    setResponse(newResponse);
  };

  async function onSubmit(input: string) {
    const newElements = [...elements];
    let base64File: string | undefined = undefined;
    let fileExtension = selectedFile?.type.split("/")[1];
    if (selectedFile) {
      base64File = await convertFileToBase64(selectedFile);
    }

    const result = await actions.agent({
      input,
      chat_history: history,
      file:
        base64File && fileExtension
          ? {
            base64: base64File,
            extension: fileExtension,
          }
          : undefined,
    });

    handleAgentResponse(result);

    newElements.push(
      <div className="flex flex-col w-full gap-1 mt-4" key={history.length}>
        {selectedFile && <FileUploadMessage file={selectedFile} />}
        <HumanMessageText content={input} />
        <div className="flex flex-col gap-1 w-full max-w-fit mr-auto mt-2">
          {result.ui}
        </div>
      </div>
    );

    setElements(newElements);
    setInput("");
    setSelectedFile(undefined);
    setHistory([...history, ["human", input], ["ai", JSON.stringify(result)]]);
  }

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [elements]);

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-md">
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {elements}
      </div>
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          if (input.trim()) {
            await onSubmit(input.trim());
          }
        }}
        className="p-4 border-t"
      >
        <div className="flex items-center space-x-2">
          <Input
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1"
          />
          <label htmlFor="file-upload" className="cursor-pointer">
            <Paperclip className="h-6 w-6 text-gray-500 hover:text-gray-700" />
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                setSelectedFile(e.target.files[0]);
              }
            }}
          />
          <Button type="submit" className="bg-blue-500 hover:bg-blue-600">
            <Send className="h-4 w-4" />
          </Button>
        </div>
        {selectedFile && (
          <p className="mt-2 text-sm text-gray-500">
            File selected: {selectedFile.name}
          </p>
        )}
      </form>
    </div>
  );
};

export default Chat;
