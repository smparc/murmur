import type { Metadata } from "next";
import "./globals.css";

// The App Router requires a root layout. Without this file Next.js cannot
// build or serve the application at all.
export const metadata: Metadata = {
  title: "Murmur — Acoustic Telemetry",
  description:
    "Real-time spatio-temporal acoustic monitoring and predictive maintenance",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased">
        {children}
      </body>
    </html>
  );
}
