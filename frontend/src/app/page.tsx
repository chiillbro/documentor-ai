// ************************* Version 1 ******************************

// frontend/app/page.tsx
"use client";

import { useState } from "react";
import { PdfUploadForm } from "@/components/ui/pdf-upload-form";
import { QueryInterface } from "@/components/ui/query-interface";
import { DocumentRead } from "@/lib/api";
import { Button } from "@/components/ui/button"; // For "Upload another document"

export default function HomePage() {
  const [uploadedDocument, setUploadedDocument] = useState<DocumentRead | null>(
    null
  );
  const [isGlobalLoading, setIsGlobalLoading] = useState(false); // Renamed for clarity

  const handleDocumentUploadSuccess = (doc: DocumentRead) => {
    setUploadedDocument(doc);
  };

  return (
    <div className="flex flex-col items-center space-y-8">
      {!uploadedDocument ? ( // Show upload form if no document is uploaded
        <section className="w-full max-w-2xl mt-10">
          <h1 className="text-4xl font-extrabold text-center mb-4 tracking-tight lg:text-5xl">
            Meet DocuMentor
          </h1>
          <p className="text-center text-lg text-muted-foreground mb-10">
            Your intelligent assistant for unlocking insights from your PDF
            documents. Upload a file to get started.
          </p>
          <PdfUploadForm
            onUploadSuccess={handleDocumentUploadSuccess}
            setIsGlobalLoading={setIsGlobalLoading}
          />
        </section>
      ) : (
        // Show query interface if a document is uploaded
        <section className="w-full max-w-4xl">
          {" "}
          {/* Wider for chat */}
          <div className="mb-6 text-center">
            <Button
              variant="outline"
              onClick={() => {
                setUploadedDocument(null); // Clear the document to go back to upload
              }}
              className="mb-4"
            >
              Upload Another Document
            </Button>
          </div>
          <QueryInterface
            documentId={uploadedDocument.id}
            documentFilename={uploadedDocument.filename} // Pass filename
            setIsGlobalLoading={setIsGlobalLoading}
          />
        </section>
      )}

      {isGlobalLoading && ( // Full-screen overlay loading state
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center z-[100]">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-primary mb-4"></div>
          <p className="text-foreground text-xl">Processing your request...</p>
        </div>
      )}
    </div>
  );
}

// ************************* Version 2 ******************************

// // frontend/app/page.tsx
// "use client"; // This page will have client-side interactions

// import { useState } from "react";
// import { PdfUploadForm } from "@/components/ui/pdf-upload-form";
// import { QueryInterface } from "@/components/ui/query-interface";
// import { DocumentRead } from "@/lib/api"; // Import the interface

// export default function HomePage() {
//   const [uploadedDocument, setUploadedDocument] = useState<DocumentRead | null>(
//     null
//   );
//   const [isLoading, setIsLoading] = useState(false);
//   const [currentView, setCurrentView] = useState<"upload" | "query">("upload");

//   // This will be passed to PdfUploadForm to update the state here
//   const handleDocumentUploadSuccess = (doc: DocumentRead) => {
//     setUploadedDocument(doc);
//     setCurrentView("query"); // Switch to query view after successful upload
//   };

//   return (
//     <div className="flex flex-col items-center">
//       {currentView === "upload" && (
//         <section className="w-full max-w-2xl">
//           <h1 className="text-3xl font-bold text-center mb-6">
//             Upload Your Document
//           </h1>
//           <p className="text-center text-muted-foreground mb-8">
//             Upload a PDF to get started. DocuMentor will help you find answers
//             within your document.
//           </p>
//           <PdfUploadForm
//             onUploadSuccess={handleDocumentUploadSuccess}
//             setIsLoading={setIsLoading}
//           />
//         </section>
//       )}

//       {currentView === "query" && uploadedDocument && (
//         <section className="w-full max-w-3xl">
//           <div className="mb-6 text-center">
//             <h2 className="text-2xl font-semibold">
//               Querying:{" "}
//               <span className="font-mono text-primary">
//                 {uploadedDocument.filename}
//               </span>
//             </h2>
//             <button
//               onClick={() => {
//                 setCurrentView("upload");
//                 setUploadedDocument(null);
//               }}
//               className="text-sm text-blue-500 hover:underline mt-2"
//             >
//               Upload another document
//             </button>
//           </div>
//           <QueryInterface
//             documentId={uploadedDocument.id} // Pass the document ID
//             setIsLoading={setIsLoading}
//           />
//         </section>
//       )}

//       {isLoading && (
//         <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
//           <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-primary"></div>
//           <p className="text-white text-xl ml-4">Processing...</p>
//         </div>
//       )}
//     </div>
//   );
// }
