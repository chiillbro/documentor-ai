// frontend/lib/api.ts
import axios from "axios";

// Define the base URL for your FastAPI backend
// In development, this will point to your Dockerized backend (localhost:8000)
// In production, this would be your deployed backend URL.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

// Define interfaces for API responses based on your Pydantic models
// (These should ideally match your backend/app/models/document_models.py)
export interface DocumentRead {
  id: string; // Assuming UUID string
  filename: string;
  content_type?: string | null;
  size?: number | null;
  uploaded_at: string; // Or Date
}

export interface QueryResponse {
  answer: string;
  source_chunks?: string[] | null;
}

export interface UploadResponse extends DocumentRead {} // Or a more specific upload response

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// // Function to upload a document
// export const uploadDocument = async (
//   file: File,
//   collectionName?: string
// ): Promise<UploadResponse> => {
//   const formData = new FormData();
//   formData.append("file", file);
//   if (collectionName) {
//     formData.append("collection_name", collectionName);
//   }

//   // For file uploads, content type should be multipart/form-data
//   const response = await apiClient.post<UploadResponse>(
//     "/documents/upload",
//     formData,
//     {
//       headers: {
//         "Content-Type": "multipart/form-data",
//       },
//     }
//   );
//   return response.data;
// };

// // Function to ask a question
// export const askQuestion = async (
//   question: string,
//   documentId?: string
// ): Promise<QueryResponse> => {
//   const payload = {
//     question,
//     document_id: documentId, // Will be undefined if not provided
//   };
//   const response = await apiClient.post<QueryResponse>(
//     "/documents/query",
//     payload
//   );
//   return response.data;
// };

export interface StreamTokenPayload {
  token: string;
}
export interface StreamSourcesPayload {
  sources: string[];
}
export interface StreamErrorPayload {
  error: string;
  detail: string;
}

// Modify existing askQuestion or create a new one for streaming
// This function will now establish an EventSource connection
// and call callbacks for different event types.
export const askQuestionStream = (
  question: string,
  documentId: string | undefined, // Make documentId explicitly string | undefined
  onToken: (payload: StreamTokenPayload) => void,
  onSources: (payload: StreamSourcesPayload) => void,
  onStreamEnd: (message: string) => void,
  onError: (payload: StreamErrorPayload) => void,
  onOpen?: () => void // Optional: callback when connection opens
  // Returns the EventSource instance so it can be closed by the caller
): EventSource => {
  const payload = {
    question,
    document_id: documentId,
  };

  // Construct URL for POST request that will be kept alive for SSE
  // We can't send a POST body directly with EventSource.
  // So, we either need a GET endpoint for streaming (passing query in URL params)
  // OR use fetch API with a readable stream (more complex to parse SSE manually)
  // OR keep POST and use fetch + ReadableStream to parse SSE.

  // For simplicity with EventSource and keeping POST, we'll use fetch and parse manually.
  // This is more robust than trying to shoehorn POST into EventSource.
  // However, a common pattern IS to use GET for EventSource.
  // Let's try GET for EventSource first, it's cleaner if backend supports it.

  // === IF BACKEND /query-stream was GET ===
  // const queryParams = new URLSearchParams();
  // queryParams.append('question', question);
  // if (documentId) queryParams.append('document_id', documentId);
  // const eventSource = new EventSource(`${API_BASE_URL}/documents/query-stream?${queryParams.toString()}`);

  // === USING FETCH FOR POST AND MANUAL SSE PARSING (More complex but keeps POST) ===
  // This is a more advanced pattern.
  // For MVP, let's assume we will refactor backend to GET for query-stream or find an SSE client that supports POST.
  // FOR NOW, let's assume a GET endpoint for streaming for simplicity of EventSource on frontend.
  // We'll need to adjust the backend route for `/query-stream` to be GET.
  // If not, we'll implement the fetch + manual parsing.

  // TEMPORARY: Assuming /query-stream is GET for EventSource demo.
  // You would need to change @router.post("/query-stream") to @router.get("/query-stream")
  // and get `query_request` from `Query()` params in FastAPI.
  // For now, to make frontend code runnable, I will proceed as if it's GET.
  // We can adjust this if GET is not feasible for your backend logic with complex QueryRequest.

  const queryParams = new URLSearchParams({ question });
  if (documentId) queryParams.set("document_id", documentId);

  const eventSourceUrl = `${API_BASE_URL}/documents/query-stream?${queryParams.toString()}`;
  const eventSource = new EventSource(eventSourceUrl);

  if (onOpen) {
    eventSource.onopen = onOpen;
  }

  eventSource.addEventListener("token", (event) => {
    const data = JSON.parse(event.data) as StreamTokenPayload;
    onToken(data);
  });

  eventSource.addEventListener("sources", (event) => {
    const data = JSON.parse(event.data) as StreamSourcesPayload;
    onSources(data);
  });

  eventSource.addEventListener("stream_end", (event) => {
    const data = JSON.parse(event.data);
    onStreamEnd(data.message || "Stream finished.");
    eventSource.close(); // Important to close the connection
  });

  eventSource.addEventListener("error", (event: MessageEvent) => {
    // EventSource's own error event (e.g. connection lost)
    // vs. our custom 'error' event from the server.
    if (event.data) {
      // Our custom error event
      try {
        const data = JSON.parse(event.data) as StreamErrorPayload;
        onError(data);
      } catch (e: any) {
        onError({
          error: "StreamParseError",
          detail: "Failed to parse error event from server.",
        });
      }
    } else {
      onError({
        error: "NetworkError",
        detail: "Connection to stream lost or could not be established.",
      });
    }
    eventSource.close();
  });

  // Custom server-sent error event
  eventSource.addEventListener("server_error", (event) => {
    // Assuming backend sends event: error
    const data = JSON.parse(event.data) as StreamErrorPayload;
    onError(data);
    eventSource.close();
  });

  // return eventSource; // Return so it can be closed
};

// Keep the non-streaming version for upload or other calls
export const uploadDocument = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await apiClient.post<UploadResponse>(
    "/documents/upload",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
    }
  );
  return response.data;
};

// You might still want a non-streaming query for some cases or for fetching sources separately
export const askQuestionNonStream = async (
  question: string,
  documentId?: string
): Promise<QueryResponse> => {
  const payload = { question, document_id: documentId };
  const response = await apiClient.post<QueryResponse>(
    "/documents/query",
    payload
  ); // Ensure you have this non-streaming endpoint
  return response.data;
};
