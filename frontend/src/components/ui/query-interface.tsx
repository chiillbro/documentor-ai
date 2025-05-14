// ************************* Version 1 ******************************

// frontend/components/query-interface.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import {
  askQuestionNonStream,
  askQuestionStream,
  StreamErrorPayload,
  StreamSourcesPayload,
  StreamTokenPayload,
} from "@/lib/api"; // DocumentRead no longer needed here
import { toast } from "sonner";
import {
  MessageSquare,
  User,
  Bot,
  Zap,
  RefreshCw,
  FileText,
} from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown"; // For rendering markdown from LLM
import remarkGfm from "remark-gfm"; // For GitHub Flavored Markdown (tables, etc.)

interface QueryInterfaceProps {
  documentId: string;
  documentFilename: string; // Pass filename for display
  setIsGlobalLoading: (loading: boolean) => void;
}

interface ChatMessage {
  id: string; // For unique key
  type: "user" | "assistant";
  text: string;
  sourceChunks?: string[] | null;
  isLoading?: boolean; // For assistant's placeholder while loading
}

// export function QueryInterface({
//   documentId,
//   documentFilename,
//   setIsGlobalLoading,
// }: QueryInterfaceProps) {
//   const [query, setQuery] = useState("");
//   const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
//   const scrollAreaRef = useRef<HTMLDivElement>(null); // Ref for scroll area viewport

//   const scrollToBottom = () => {
//     if (scrollAreaRef.current) {
//       const scrollViewport = scrollAreaRef.current
//         .children[1] as HTMLDivElement; // This is a bit hacky, depends on ScrollArea internals
//       if (scrollViewport) {
//         scrollViewport.scrollTop = scrollViewport.scrollHeight;
//       }
//     }
//   };

//   useEffect(() => {
//     scrollToBottom();
//   }, [chatHistory]);

//   const handleSubmitQuery = async (e?: React.FormEvent<HTMLFormElement>) => {
//     if (e) e.preventDefault();
//     const currentQuery = query.trim();
//     if (!currentQuery) {
//       toast.error("Please enter a question.");
//       return;
//     }

//     setIsGlobalLoading(true); // Global loading for entire screen dim
//     const userMessage: ChatMessage = {
//       id: crypto.randomUUID(),
//       type: "user",
//       text: currentQuery,
//     };
//     const assistantLoadingMessage: ChatMessage = {
//       id: crypto.randomUUID(),
//       type: "assistant",
//       text: "",
//       isLoading: true,
//     };

//     setChatHistory((prev) => [...prev, userMessage, assistantLoadingMessage]);
//     setQuery(""); // Clear input immediately

//     try {
//       const response = await askQuestion(userMessage.text, documentId);
//       const assistantMessage: ChatMessage = {
//         id: assistantLoadingMessage.id, // Use the same ID to replace the loading one
//         type: "assistant",
//         text: response.answer,
//         sourceChunks: response.source_chunks,
//         isLoading: false,
//       };
//       // Replace loading message with actual response
//       setChatHistory((prev) =>
//         prev.map((msg) =>
//           msg.id === assistantLoadingMessage.id ? assistantMessage : msg
//         )
//       );
//     } catch (error: any) {
//       console.error("Query failed:", error);
//       const detail =
//         error?.response?.data?.detail ||
//         error.message ||
//         "An unknown error occurred.";
//       toast.error(`Query failed: ${detail}`);
//       const errorMessage: ChatMessage = {
//         id: assistantLoadingMessage.id,
//         type: "assistant",
//         text: `Sorry, I encountered an error: ${detail}`,
//         isLoading: false,
//       };
//       setChatHistory((prev) =>
//         prev.map((msg) =>
//           msg.id === assistantLoadingMessage.id ? errorMessage : msg
//         )
//       );
//     } finally {
//       setIsGlobalLoading(false);
//     }
//   };

//   return (
//     <div className="flex flex-col bg-card border rounded-lg shadow-xl">
//       <div className="p-4 border-b">
//         <h3 className="font-semibold text-lg">
//           Chat with:{" "}
//           <span className="font-mono text-sm text-primary">
//             {documentFilename}
//           </span>
//         </h3>
//       </div>

//       <ScrollArea
//         className="flex-grow p-4 h-[calc(100vh-22rem)] max-h-[600px]"
//         ref={scrollAreaRef}
//       >
//         {chatHistory.length === 0 && (
//           <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
//             <MessageSquare size={48} className="mb-4 opacity-50" />
//             <p className="text-lg">Ask a question about your document.</p>
//           </div>
//         )}
//         <div className="space-y-6">
//           {chatHistory.map((message) => (
//             <div
//               key={message.id}
//               className={cn(
//                 "flex items-end space-x-2",
//                 message.type === "user" ? "justify-end" : "justify-start"
//               )}
//             >
//               {message.type === "assistant" && (
//                 <Bot className="h-7 w-7 text-primary flex-shrink-0 mb-1" />
//               )}
//               <Card
//                 className={cn(
//                   "max-w-[80%] p-3 text-sm",
//                   message.type === "user"
//                     ? "bg-primary text-primary-foreground rounded-tr-none"
//                     : "bg-muted text-foreground rounded-tl-none",
//                   message.isLoading && "animate-pulse"
//                 )}
//               >
//                 <CardContent className="p-0 whitespace-pre-wrap">
//                   {message.isLoading ? (
//                     <div className="space-y-2">
//                       <div className="h-2.5 bg-foreground/20 rounded-full w-32"></div>
//                       <div className="h-2.5 bg-foreground/20 rounded-full w-48"></div>
//                       <div className="h-2.5 bg-foreground/20 rounded-full w-40"></div>
//                     </div>
//                   ) : (
//                     <ReactMarkdown remarkPlugins={[remarkGfm]}>
//                       {message.text}
//                     </ReactMarkdown>
//                   )}
//                   {message.type === "assistant" &&
//                     !message.isLoading &&
//                     message.sourceChunks &&
//                     message.sourceChunks.length > 0 && (
//                       <details className="mt-3 text-xs text-muted-foreground cursor-pointer group">
//                         <summary className="group-hover:text-primary transition-colors">
//                           View Sources ({message.sourceChunks.length})
//                         </summary>
//                         <ScrollArea className="max-h-40 mt-1.5 p-2 border rounded bg-background text-xs">
//                           <div className="space-y-2">
//                             {message.sourceChunks.map((chunk, chunkIndex) => (
//                               <div
//                                 key={chunkIndex}
//                                 className="p-2 border-b last:border-b-0 hover:bg-muted/50 rounded-sm"
//                               >
//                                 <p className="line-clamp-3">{chunk}</p>
//                               </div>
//                             ))}
//                           </div>
//                         </ScrollArea>
//                       </details>
//                     )}
//                 </CardContent>
//               </Card>
//               {message.type === "user" && (
//                 <User className="h-7 w-7 text-muted-foreground flex-shrink-0 mb-1" />
//               )}
//             </div>
//           ))}
//         </div>
//         <ScrollBar orientation="vertical" />
//       </ScrollArea>

//       <div className="p-4 border-t bg-background">
//         <form
//           onSubmit={handleSubmitQuery}
//           className="flex items-center space-x-2"
//         >
//           <Textarea
//             value={query}
//             onChange={(e) => setQuery(e.target.value)}
//             placeholder="Type your question here..."
//             className="flex-grow resize-none text-sm min-h-[40px]"
//             rows={1} // Start with 1 row, it can expand
//             onKeyDown={(e) => {
//               if (e.key === "Enter" && !e.shiftKey && !isChatHistoryLoading()) {
//                 e.preventDefault();
//                 handleSubmitQuery();
//               }
//             }}
//             disabled={isChatHistoryLoading()}
//           />
//           <Button
//             type="submit"
//             size="icon"
//             className="h-10 w-10"
//             disabled={isChatHistoryLoading() || !query.trim()}
//           >
//             {isChatHistoryLoading() ? (
//               <RefreshCw className="h-5 w-5 animate-spin" />
//             ) : (
//               <Zap className="h-5 w-5" />
//             )}
//             <span className="sr-only">Ask</span>
//           </Button>
//         </form>
//       </div>
//     </div>
//   );

//   function isChatHistoryLoading() {
//     return chatHistory.some((msg) => msg.isLoading);
//   }
// }

export function QueryInterface({
  documentId,
  documentFilename,
  setIsGlobalLoading,
}: QueryInterfaceProps) {
  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  // const [isResponding, setIsResponding] = useState(false); // Replaced by isLoading state on assistant message
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const currentEventSource = useRef<EventSource | null>(null); // To store the EventSource instance

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current
        .children[1] as HTMLDivElement; // This is a bit hacky, depends on ScrollArea internals
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  };
  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleSubmitQuery = async (e?: React.FormEvent<HTMLFormElement>) => {
    if (e) e.preventDefault();
    const currentQuery = query.trim();
    if (!currentQuery) {
      toast.error("Please enter a question.");
      return;
    }

    // Abort previous stream if any
    if (currentEventSource.current) {
      currentEventSource.current.close();
    }

    setIsGlobalLoading(true);
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      type: "user",
      text: currentQuery,
    };
    const assistantMessageId = crypto.randomUUID();
    const initialAssistantMessage: ChatMessage = {
      id: assistantMessageId,
      type: "assistant",
      text: "", // Start with empty text
      isLoading: true,
    };

    setChatHistory((prev) => [...prev, userMessage, initialAssistantMessage]);
    setQuery("");

    let accumulatedText = "";

    currentEventSource.current = askQuestionStream(
      userMessage.text,
      documentId,
      (payload: StreamTokenPayload) => {
        // onToken
        accumulatedText += payload.token;
        setChatHistory((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, text: accumulatedText, isLoading: true }
              : msg
          )
        );
      },
      (payload: StreamSourcesPayload) => {
        // onSources
        setChatHistory((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, sourceChunks: payload.sources }
              : msg
          )
        );
        // You might want to update isLoading to false here if sources are the last thing
      },
      (message: string) => {
        // onStreamEnd
        setChatHistory((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId ? { ...msg, isLoading: false } : msg
          )
        );
        toast.info(message || "Stream finished.");
        setIsGlobalLoading(false);
        if (currentEventSource.current) currentEventSource.current.close();
      },
      (payload: StreamErrorPayload) => {
        // onError
        toast.error(`Error: ${payload.detail || payload.error}`);
        setChatHistory((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, text: `Error: ${payload.detail}`, isLoading: false }
              : msg
          )
        );
        setIsGlobalLoading(false);
        if (currentEventSource.current) currentEventSource.current.close();
      },
      () => {
        console.info("SSE Connection Opened");
        // onOpen (optional)
        // logger.info("SSE Connection Opened"); // Use a frontend logger if you have one
      }
    );
  };

  // Cleanup EventSource on component unmount
  useEffect(() => {
    return () => {
      if (currentEventSource.current) {
        currentEventSource.current.close();
      }
    };
  }, []);

  // Helper to check if the last assistant message is still loading
  const isAssistantResponding = () => {
    const lastMessage = chatHistory[chatHistory.length - 1];
    return lastMessage?.type === "assistant" && lastMessage.isLoading;
  };

  return (
    <div className="flex flex-col bg-card border rounded-lg shadow-xl">
      {/* ... (header part remains similar) ... */}

      <div className="p-4 border-b">
        <h3 className="font-semibold text-lg">
          Chat with:{" "}
          <span className="font-mono text-sm text-primary">
            {documentFilename}
          </span>
        </h3>
      </div>

      <ScrollArea
        className="flex-grow p-4 h-[calc(100vh-22rem)] max-h-[600px]"
        ref={scrollAreaRef}
      >
        {chatHistory.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <MessageSquare size={48} className="mb-4 opacity-50" />
            <p className="text-lg">Ask a question about your document.</p>
          </div>
        )}
        <div>
          {/* Ensure ReactMarkdown is used for message.text in the map */}
          {chatHistory.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex items-end space-x-2 mb-4", // Added mb-4 for spacing
                message.type === "user" ? "justify-end" : "justify-start"
              )}
            >
              {message.type === "assistant" && (
                <Bot className="h-7 w-7 text-primary flex-shrink-0 self-start mt-1" /> // Align icon better
              )}
              <Card
                className={cn(
                  "max-w-[80%] p-3 text-sm shadow-md", // Added shadow
                  message.type === "user"
                    ? "bg-primary text-primary-foreground rounded-tr-none"
                    : "bg-muted text-foreground rounded-tl-none"
                )}
              >
                <CardContent className="p-0">
                  {message.isLoading ? (
                    <div className="space-y-1.5 py-1">
                      {" "}
                      {/* Adjusted pulsing animation */}
                      <div className="h-2.5 bg-foreground/10 rounded-full w-32 animate-pulse"></div>
                      <div className="h-2.5 bg-foreground/10 rounded-full w-48 animate-pulse delay-75"></div>
                      <div className="h-2.5 bg-foreground/10 rounded-full w-40 animate-pulse delay-150"></div>
                    </div>
                  ) : (
                    <ReactMarkdown
                      // className="prose prose-sm dark:prose-invert max-w-none" // Basic prose styling
                      remarkPlugins={[remarkGfm]}
                      components={{
                        // Custom renderers for better control if needed
                        p: ({ node, ...props }) => (
                          <p className="mb-1 last:mb-0" {...props} />
                        ),
                        // Add more custom renderers for ul, ol, code etc. for enterprise look
                      }}
                    >
                      {message.text}
                    </ReactMarkdown>
                  )}
                  {message.type === "assistant" &&
                    !message.isLoading &&
                    message.sourceChunks &&
                    message.sourceChunks.length > 0 && (
                      // ... (source chunks display remains similar) ...
                      <details className="mt-3 text-xs text-muted-foreground cursor-pointer group">
                        <summary className="group-hover:text-primary transition-colors font-medium">
                          View Sources ({message.sourceChunks.length})
                        </summary>
                        <ScrollArea className="max-h-40 mt-1.5 p-2 border rounded bg-background/70 text-xs">
                          <div className="space-y-2">
                            {message.sourceChunks.map((chunk, chunkIndex) => (
                              <div
                                key={chunkIndex}
                                className="p-2 border-b border-border/50 last:border-b-0 hover:bg-muted/50 rounded-sm transition-colors"
                              >
                                <p className="line-clamp-3 text-foreground/80">
                                  {chunk}
                                </p>
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </details>
                    )}
                </CardContent>
              </Card>
              {message.type === "user" && (
                <User className="h-7 w-7 text-muted-foreground flex-shrink-0 self-start mt-1" /> // Align icon better
              )}
            </div>
          ))}
        </div>
        <ScrollBar orientation="vertical" />
      </ScrollArea>

      <div className="p-4 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <form
          onSubmit={handleSubmitQuery}
          className="flex items-center space-x-2"
        >
          <Textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask anything about the document..."
            className="flex-grow resize-none text-sm min-h-[40px] max-h-[150px] rounded-md shadow-sm"
            rows={1}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !isAssistantResponding()
              ) {
                e.preventDefault();
                handleSubmitQuery();
              }
            }}
            disabled={isAssistantResponding()}
          />
          <Button
            type="submit"
            size="icon"
            className="h-10 w-10 flex-shrink-0 rounded-md shadow-sm"
            disabled={isAssistantResponding() || !query.trim()}
          >
            {isAssistantResponding() ? (
              <RefreshCw className="h-5 w-5 animate-spin" />
            ) : (
              <Zap className="h-5 w-5" />
            )}
            <span className="sr-only">Ask</span>
          </Button>
        </form>
      </div>
    </div>
  );
}

// ************************* Version 2 ******************************

// // frontend/components/query-interface.tsx
// "use client";

// import { useState } from "react";
// import { Button } from "@/components/ui/button";
// import { Textarea } from "@/components/ui/textarea";
// import { ScrollArea } from "@/components/ui/scroll-area";
// import {
//   Card,
//   CardContent,
//   CardDescription,
//   CardHeader,
//   CardTitle,
// } from "@/components/ui/card";
// import { askQuestion, QueryResponse } from "@/lib/api";
// import { toast } from "sonner";
// import { MessageSquare, Zap, RefreshCw } from "lucide-react"; // Example icons

// interface QueryInterfaceProps {
//   documentId: string; // Or the identifier your backend expects
//   setIsLoading: (loading: boolean) => void;
// }

// interface ChatMessage {
//   type: "user" | "assistant";
//   text: string;
//   sourceChunks?: string[];
// }

// export function QueryInterface({
//   documentId,
//   setIsLoading,
// }: QueryInterfaceProps) {
//   const [query, setQuery] = useState("");
//   const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
//   const [isResponding, setIsResponding] = useState(false);

//   const handleSubmitQuery = async (e?: React.FormEvent<HTMLFormElement>) => {
//     if (e) e.preventDefault();
//     if (!query.trim()) {
//       toast.error("Please enter a question.");
//       return;
//     }

//     setIsLoading(true);
//     setIsResponding(true);
//     const userMessage: ChatMessage = { type: "user", text: query };
//     setChatHistory((prev) => [...prev, userMessage]);
//     setQuery(""); // Clear input

//     try {
//       const response = await askQuestion(userMessage.text, documentId);
//       const assistantMessage: ChatMessage = {
//         type: "assistant",
//         text: response.answer,
//         sourceChunks: response.source_chunks,
//       };
//       setChatHistory((prev) => [...prev, assistantMessage]);
//     } catch (error) {
//       console.error("Query failed:", error);
//       const errorMessage = `Query failed. ${
//         (error as any)?.response?.data?.detail || (error as Error).message
//       }`;
//       toast.error(errorMessage);
//       setChatHistory((prev) => [
//         ...prev,
//         { type: "assistant", text: `Error: ${errorMessage}` },
//       ]);
//     } finally {
//       setIsLoading(false);
//       setIsResponding(false);
//     }
//   };

//   return (
//     <div className="flex flex-col">
//       {" "}
//       {/* Adjust height as needed */}
//       <ScrollArea className="flex-grow p-4 border rounded-md mb-4 bg-muted/30 h-[calc(100vh-12rem)] max-h-[700px]">
//         {chatHistory.length === 0 && (
//           <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
//             <MessageSquare size={48} className="mb-4" />
//             <p>Ask a question about the document.</p>
//           </div>
//         )}
//         <div className="space-y-4">
//           {chatHistory.map((message, index) => (
//             <div
//               key={index}
//               className={`flex ${
//                 message.type === "user" ? "justify-end" : "justify-start"
//               }`}
//             >
//               <Card
//                 className={`max-w-[75%] p-3 ${
//                   message.type === "user"
//                     ? "bg-primary text-primary-foreground rounded-br-none"
//                     : "bg-card text-card-foreground rounded-bl-none"
//                 }`}
//               >
//                 <CardContent className="p-0 text-sm whitespace-pre-wrap">
//                   {message.text}
//                   {message.type === "assistant" &&
//                     message.sourceChunks &&
//                     message.sourceChunks.length > 0 && (
//                       <details className="mt-2 text-xs opacity-70">
//                         <summary className="cursor-pointer">
//                           View Sources ({message.sourceChunks.length})
//                         </summary>
//                         <ScrollArea className="max-h-32 mt-1 p-2 border rounded bg-background/50">
//                           {message.sourceChunks.map((chunk, chunkIndex) => (
//                             <p
//                               key={chunkIndex}
//                               className="mb-1 border-b pb-1 last:border-b-0 last:pb-0"
//                             >
//                               {chunk.length > 150
//                                 ? chunk.substring(0, 150) + "..."
//                                 : chunk}
//                             </p>
//                           ))}
//                         </ScrollArea>
//                       </details>
//                     )}
//                 </CardContent>
//               </Card>
//             </div>
//           ))}
//         </div>
//       </ScrollArea>
//       <form onSubmit={handleSubmitQuery} className="flex items-start space-x-2">
//         <Textarea
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//           placeholder="Ask something about the document..."
//           className="flex-grow resize-none"
//           rows={2}
//           onKeyDown={(e) => {
//             if (e.key === "Enter" && !e.shiftKey) {
//               e.preventDefault();
//               handleSubmitQuery();
//             }
//           }}
//           disabled={isResponding}
//         />
//         <Button
//           type="submit"
//           size="lg"
//           disabled={isResponding || !query.trim()}
//         >
//           {isResponding ? (
//             <RefreshCw className="h-5 w-5 animate-spin" />
//           ) : (
//             <Zap className="h-5 w-5" />
//           )}
//           <span className="ml-2 sm:inline hidden">Ask</span>
//         </Button>
//       </form>
//     </div>
//   );
// }
