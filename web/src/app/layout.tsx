import type { Metadata, Viewport } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Suspense } from "react";
import { Providers } from "@/components/providers";
import { Shell } from "@/components/shell";
import "./globals.css";

// Root layout: wires fonts, wraps every route in Providers (react-query +
// theme, components/providers.tsx) and the persistent Shell chrome
// (components/shell.tsx). The Suspense fallback covers the brief window
// while Shell reads URL search params via hooks/use-context.ts.
export const metadata: Metadata = {
  title: "TimesFM · Forecasting workspace",
  description:
    "A local analytical workspace for TimesFM-3 forecasting, experiments, and tracking.",
};
export const viewport: Viewport = { width: "device-width", initialScale: 1 };
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${GeistSans.variable} ${GeistMono.variable}`}>
        <Providers>
          <Suspense
            fallback={<div className="p-8 text-sm">Opening workspace…</div>}
          >
            <Shell>{children}</Shell>
          </Suspense>
        </Providers>
      </body>
    </html>
  );
}
