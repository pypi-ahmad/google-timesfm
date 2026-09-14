"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { useState, type ReactNode } from "react";
import { ApiError } from "@/lib/api";

// Root client providers: a single QueryClient instance (react-query) plus
// the color-theme provider, wrapping the Shell in app/layout.tsx.
export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 15_000,
            // Retry once, but only for failures that aren't a client error
            // (status < 500) — a 4xx (bad input, not found) won't succeed
            // on retry, so only treat likely-transient (network/5xx)
            // failures as worth retrying.
            retry: (count, error) =>
              !(error instanceof ApiError && error.status < 500) && count < 1,
            refetchOnWindowFocus: true,
          },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <ThemeProvider
        attribute="class"
        defaultTheme="light"
        enableSystem
        disableTransitionOnChange
      >
        {children}
      </ThemeProvider>
    </QueryClientProvider>
  );
}
