// frontend/app/layout.tsx
import type { Metadata } from "next";
import { Inter as FontSans } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import { Toaster } from "@/components/ui/sonner"; // Or react-hot-toast's Toaster
// import { ThemeProvider } from "@/components/theme-provider"; // For V2
import { SiteHeader } from "@/components/ui/site-header";

const fontSans = FontSans({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "DocuMentor AI",
  description: "A context-aware document assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={cn(
          "min-h-screen bg-background font-sans antialiased",
          fontSans.variable
        )}
      >
        {/* <ThemeProvider attribute="class" defaultTheme="system" enableSystem> // For V2 */}
        <div className="relative flex min-h-screen flex-col">
          <SiteHeader />
          <main className="flex-1 container mx-auto px-4 py-8">{children}</main>
          {/* Add a footer later if desired */}
        </div>
        <Toaster richColors position="top-right" />
        {/* </ThemeProvider> */}
      </body>
    </html>
  );
}
