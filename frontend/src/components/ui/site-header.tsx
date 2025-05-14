// frontend/components/site-header.tsx
import Link from "next/link";
import { BrainCircuit } from "lucide-react"; // Example icon

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 max-w-screen-2xl items-center">
        <Link href="/" className="mr-6 flex items-center space-x-2 pl-4">
          <BrainCircuit className="h-6 w-6 text-primary" />
          <span className="font-bold sm:inline-block text-lg">DocuMentor</span>
        </Link>
        {/* Add navigation items later if needed */}
      </div>
    </header>
  );
}
