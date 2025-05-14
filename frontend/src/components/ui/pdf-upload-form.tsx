// ************************* Version 2 ******************************

// frontend/components/pdf-upload-form.tsx
"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import {
  UploadCloud,
  FileText,
  X,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import { uploadDocument, DocumentRead } from "../../lib/api";

interface PdfUploadFormProps {
  onUploadSuccess: (document: DocumentRead) => void;
  setIsGlobalLoading: (loading: boolean) => void; // Changed name for clarity
}

export function PdfUploadForm({
  onUploadSuccess,
  setIsGlobalLoading,
}: PdfUploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  // We won't use uploadProgress for now as it's simulated and backend doesn't provide real progress

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0];
      // Dropzone's accept prop already filters, but an extra check doesn't hurt
      if (selectedFile.type === "application/pdf") {
        setFile(selectedFile);
        toast.info(`Selected: ${selectedFile.name}`);
      } else {
        toast.error("Invalid file type. Please upload a PDF.");
      }
    } else if (rejectedFiles && rejectedFiles.length > 0) {
      toast.error("File rejected. Please ensure it's a PDF.");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } =
    useDropzone({
      onDrop,
      accept: { "application/pdf": [".pdf"] },
      multiple: false,
      maxSize: 25 * 1024 * 1024, // Example: 25MB limit
      onDropRejected: (fileRejections) => {
        fileRejections.forEach(({ file, errors }) => {
          errors.forEach((err) => {
            if (err.code === "file-too-large") {
              toast.error(
                `Error: ${file.name} is too large. Max size is 25MB.`
              );
            } else if (err.code === "file-invalid-type") {
              toast.error(`Error: ${file.name} is not a PDF file.`);
            } else {
              toast.error(`Error with ${file.name}: ${err.message}`);
            }
          });
        });
      },
    });

  const handleUpload = async () => {
    if (!file) {
      toast.error("Please select a PDF file to upload.");
      return;
    }

    setIsGlobalLoading(true);
    setIsUploading(true);

    const toastId = toast.loading(`Uploading "${file.name}"...`, {
      icon: <UploadCloud className="animate-pulse" />,
    });

    try {
      const uploadedDoc = await uploadDocument(file); // API call
      toast.success(`"${uploadedDoc.filename}" processed successfully!`, {
        id: toastId,
        icon: <CheckCircle className="text-green-500" />,
      });
      onUploadSuccess(uploadedDoc); // Pass the full document object
    } catch (error: any) {
      console.error("Upload failed:", error);
      const detail =
        error?.response?.data?.detail ||
        error.message ||
        "An unknown error occurred.";
      toast.error(`Upload failed: ${detail}`, {
        id: toastId,
        icon: <AlertTriangle className="text-red-500" />,
      });
    } finally {
      setIsGlobalLoading(false);
      setIsUploading(false);
      // Do not clear the file here, let HomePage manage it or add a clear button
    }
  };

  // Update file state if acceptedFiles changes (e.g., user selects a new file after an error)
  // This handles the case where onDrop was called but we might have had an internal error.
  // Or if a file was dragged after one was already selected.
  if (acceptedFiles && acceptedFiles.length > 0 && acceptedFiles[0] !== file) {
    if (acceptedFiles[0].type === "application/pdf") {
      setFile(acceptedFiles[0]);
    }
  }

  return (
    <div className="space-y-6 p-6 border rounded-lg shadow-lg bg-card">
      <div
        {...getRootProps()}
        className={`p-10 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors
                    ${
                      isDragActive
                        ? "border-primary bg-primary/10 ring-2 ring-primary"
                        : "border-muted-foreground/50 hover:border-primary/70"
                    }`}
      >
        <input {...getInputProps()} />
        <UploadCloud className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
        {isDragActive ? (
          <p className="text-lg font-semibold text-primary">
            Drop the PDF here ...
          </p>
        ) : (
          <>
            <p className="text-lg font-semibold text-foreground">
              Drag & drop PDF, or click to select
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Max file size: 25MB
            </p>
          </>
        )}
      </div>

      {file && (
        <div className="p-3 border rounded-md bg-muted/50 flex items-center justify-between animate-in fade-in-50 duration-300">
          <div className="flex items-center space-x-3 overflow-hidden">
            <FileText className="h-7 w-7 text-primary flex-shrink-0" />
            <div className="truncate">
              <p className="font-medium text-sm truncate" title={file.name}>
                {file.name}
              </p>
              <p className="text-xs text-muted-foreground">
                ({(file.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setFile(null)}
            disabled={isUploading}
            aria-label="Remove file"
          >
            <X className="h-5 w-5 text-muted-foreground hover:text-destructive" />
          </Button>
        </div>
      )}

      <Button
        onClick={handleUpload}
        disabled={!file || isUploading}
        className="w-full text-base py-6"
        size="lg"
      >
        {isUploading ? (
          <>
            <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <UploadCloud className="mr-2 h-5 w-5" />
            Upload and Process Document
          </>
        )}
      </Button>
    </div>
  );
}

// ************************* Version 1 ******************************

// // frontend/components/pdf-upload-form.tsx
// "use client";

// import { useState, useCallback } from "react";
// import { useDropzone } from "react-dropzone";
// import { Button } from "@/components/ui/button";
// import { Progress } from "@/components/ui/progress"; // Using Shadcn progress
// import { toast } from "sonner"; // Using Shadcn sonner
// import { UploadCloud, FileText, X } from "lucide-react";
// import { uploadDocument, DocumentRead } from "@/lib/api";

// interface PdfUploadFormProps {
//   onUploadSuccess: (document: DocumentRead) => void;
//   setIsLoading: (loading: boolean) => void;
// }

// export function PdfUploadForm({
//   onUploadSuccess,
//   setIsLoading,
// }: PdfUploadFormProps) {
//   const [file, setFile] = useState<File | null>(null);
//   const [uploadProgress, setUploadProgress] = useState(0); // For actual progress later

//   const onDrop = useCallback((acceptedFiles: File[]) => {
//     if (acceptedFiles && acceptedFiles.length > 0) {
//       const selectedFile = acceptedFiles[0];
//       if (selectedFile.type === "application/pdf") {
//         setFile(selectedFile);
//       } else {
//         toast.error("Invalid file type. Please upload a PDF.");
//       }
//     }
//   }, []);

//   const { getRootProps, getInputProps, isDragActive } = useDropzone({
//     onDrop,
//     accept: { "application/pdf": [".pdf"] },
//     multiple: false,
//   });

//   const handleUpload = async () => {
//     if (!file) {
//       toast.error("Please select a PDF file to upload.");
//       return;
//     }

//     setIsLoading(true);
//     setUploadProgress(0); // Reset progress

//     try {
//       // Simulate progress for now, replace with actual progress tracking if backend supports it
//       // For Axios, you can use onUploadProgress config
//       let currentProgress = 0;
//       const interval = setInterval(() => {
//         currentProgress += 10;
//         if (currentProgress <= 100) {
//           setUploadProgress(currentProgress);
//         } else {
//           clearInterval(interval);
//         }
//       }, 200); // Simulate progress update every 200ms

//       const uploadedDoc = await uploadDocument(file);
//       clearInterval(interval); // Clear simulation
//       setUploadProgress(100);
//       toast.success(`"${uploadedDoc.filename}" uploaded successfully!`);
//       onUploadSuccess(uploadedDoc);
//     } catch (error) {
//       console.error("Upload failed:", error);
//       toast.error(
//         `Upload failed. ${
//           (error as any)?.response?.data?.detail || (error as Error).message
//         }`
//       );
//       setUploadProgress(0);
//     } finally {
//       setIsLoading(false);
//       // setFile(null); // Optionally clear the file after upload
//     }
//   };

//   return (
//     <div className="space-y-6">
//       <div
//         {...getRootProps()}
//         className={`p-8 border-2 border-dashed rounded-lg text-center cursor-pointer
//                     ${
//                       isDragActive
//                         ? "border-primary bg-primary/10"
//                         : "border-muted-foreground hover:border-primary"
//                     }`}
//       >
//         <input {...getInputProps()} />
//         <UploadCloud className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
//         {isDragActive ? (
//           <p className="text-primary">Drop the PDF here ...</p>
//         ) : (
//           <p className="text-muted-foreground">
//             Drag & drop a PDF file here, or click to select file
//           </p>
//         )}
//       </div>

//       {file && (
//         <div className="p-4 border rounded-lg bg-muted flex items-center justify-between">
//           <div className="flex items-center space-x-2">
//             <FileText className="h-6 w-6 text-primary" />
//             <span>{file.name}</span>
//             <span className="text-sm text-muted-foreground">
//               ({(file.size / 1024 / 1024).toFixed(2)} MB)
//             </span>
//           </div>
//           <Button
//             variant="ghost"
//             size="icon"
//             onClick={() => {
//               setFile(null);
//               setUploadProgress(0);
//             }}
//           >
//             <X className="h-4 w-4" />
//           </Button>
//         </div>
//       )}

//       {file && uploadProgress > 0 && (
//         <Progress value={uploadProgress} className="w-full" />
//       )}

//       <Button
//         onClick={handleUpload}
//         disabled={!file || (uploadProgress > 0 && uploadProgress < 100)}
//         className="w-full"
//         size="lg"
//       >
//         <UploadCloud className="mr-2 h-5 w-5" />
//         Upload and Process Document
//       </Button>
//     </div>
//   );
// }
