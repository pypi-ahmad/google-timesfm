import Link from "next/link";
export default function NotFound() {
  return (
    <div className="panel p-8">
      <h1 className="text-lg font-semibold">Page not found</h1>
      <p className="mt-3 text-sm text-muted-foreground">
        <Link href="/overview" className="text-primary underline">
          Return to your workspace
        </Link>
      </p>
    </div>
  );
}
