"use client";
import { Button } from "@/components/ui/button";
// Next.js route-segment error boundary. `reset()` re-renders the segment
// in place rather than a full reload; the error's own message/detail is
// intentionally not shown here to avoid leaking backend error content to
// the user — see lib/api.ts ApiError for what's normally surfaced instead.
export default function ErrorPage({
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="panel mx-auto max-w-lg p-8">
      <h1 className="text-lg font-semibold">
        This page could not be displayed
      </h1>
      <p className="my-3 text-sm leading-6 text-muted-foreground">
        Reload the page to reconnect. Submitted jobs and saved runs remain
        available on the server.
      </p>
      <Button onClick={reset}>Try again</Button>
    </div>
  );
}
