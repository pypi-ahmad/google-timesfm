import { OverviewPage } from "@/features/overview-page";
// Root "/" route: renders the Overview page directly. Every other named
// route is handled by the app/[page]/page.tsx catch-all.
export default function Page() {
  return <OverviewPage />;
}
